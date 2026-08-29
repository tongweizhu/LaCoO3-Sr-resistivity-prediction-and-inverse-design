"""Browser-first model fine-tuning workflow.

User uploads are held only in the active Rio session.  The existing training
routine still expects paths, so this page writes them to a ``TemporaryDirectory``
for the duration of the training call and reads all downloadable artifacts back
into memory before that directory is removed.
"""

from __future__ import annotations

import asyncio
from dataclasses import field
from pathlib import Path
import tempfile
from typing import Any

import pandas as pd
import rio
from rio.components.class_container import ClassContainer

from ..components.palette import FIG19_COOL_PALE, FIG19_DEEP_BLUE
from ..components.ui import PageShell, SectionCard, StatusBanner
from ..workflow_utils import (
    FineTuneValidation,
    ModelInspection,
    dataframe_to_xlsx_bytes,
    inspect_model_bytes,
    read_dataframe_from_bytes,
    validate_fine_tune_frame,
)


@rio.page(
    name="Model Fine-tuning",
    url_segment="finetune",
    icon="material/tune",
    order=40,
)
class FineTunePage(rio.Component):
    """Fine-tune a trusted joblib package without exposing server paths."""

    status_text: str = "Upload a trusted model and new data; structural and numeric preflight runs first."
    status_style: str = "info"

    model_name: str = ""
    data_name: str = ""
    model_type: str = ""
    model_features: list[str] = []
    category_mappings: dict[str, dict[str, float]] = {}
    input_transforms: dict[str, str] = {}
    conditional_missing_when_a_zero: list[str] = []
    data_row_count: int = 0
    target_name: str = ""
    extra_feature_columns: list[str] = []

    preflight_ready: bool = False
    preflight_summary: str = "Upload a model and data to begin."
    num_rounds: str = "50"
    learning_rate: str = "0.01"
    is_running: bool = False

    last_summary: dict[str, Any] = {}
    has_results: bool = False

    # Large/unsafe objects are never serialized as component state.
    _model_bytes: bytes = field(default=b"", init=False, repr=False)
    _data_bytes: bytes = field(default=b"", init=False, repr=False)
    _data_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _last_validation: FineTuneValidation | None = field(default=None, init=False, repr=False)
    _fine_tuned_model_bytes: bytes = field(default=b"", init=False, repr=False)
    _comparison_csv_bytes: bytes = field(default=b"", init=False, repr=False)
    _metrics_xlsx_bytes: bytes = field(default=b"", init=False, repr=False)

    def _set_status(self, text: str, style: str = "info") -> None:
        self.status_text = text
        self.status_style = style

    def _refresh_preflight(self) -> None:
        """Recalculate the preflight whenever the selected model/data changes."""

        self.preflight_ready = False
        self._last_validation = None
        self.extra_feature_columns = []
        self.target_name = ""

        if not self._model_bytes and not self._data_bytes:
            self.preflight_summary = "Upload a model and data to begin."
            return
        if not self._model_bytes:
            self.preflight_summary = "New data loaded; upload the base model (.joblib)."
            return
        if self._data_frame is None:
            self.preflight_summary = "Base model loaded; upload new data (CSV or XLSX)."
            return

        validation = validate_fine_tune_frame(
            self._data_frame,
            tuple(self.model_features),
            self.category_mappings,
            self.input_transforms,
            self.conditional_missing_when_a_zero,
        )
        self._last_validation = validation
        self.target_name = validation.target_name
        self.extra_feature_columns = list(validation.frame.extras)
        self.preflight_summary = self._preflight_message(validation)
        self.preflight_ready = validation.is_valid

    @staticmethod
    def _preflight_message(validation: FineTuneValidation) -> str:
        issues: list[str] = []
        frame = validation.frame
        if frame.missing:
            issues.append("missing model fields: " + ", ".join(frame.missing))
        if frame.duplicates:
            issues.append("duplicate fields: " + ", ".join(frame.duplicates))

        for feature, count in frame.empty_cells.items():
            if count:
                issues.append(f"{feature}: {count} empty")
        for feature, count in frame.non_numeric_cells.items():
            if count:
                issues.append(f"{feature}: {count} non-numeric")
        for feature, count in frame.non_finite_cells.items():
            if count:
                issues.append(f"{feature}: {count} non-finite")
        for feature, count in frame.invalid_category_cells.items():
            if count:
                issues.append(f"{feature}: {count} invalid categories")
        for feature, count in frame.invalid_binary_cells.items():
            if count:
                issues.append(f"{feature}: {count} values other than 0 or 1")
        for feature, count in frame.non_positive_transform_cells.items():
            if count:
                issues.append(f"{feature}: {count} values must be greater than 0")

        if validation.row_count < 2:
            issues.append("at least 2 valid samples are required")
        if validation.target_empty_cells:
            issues.append(f"target column: {validation.target_empty_cells} empty")
        if validation.target_non_numeric_cells:
            issues.append(f"target column: {validation.target_non_numeric_cells} non-numeric")
        if validation.target_non_positive_cells:
            issues.append(f"target column: {validation.target_non_positive_cells} non-positive or non-finite")

        if issues:
            return "Preflight failed: " + "; ".join(issues) + "."

        target = validation.target_name or "last column"
        return (
            f"Preflight passed: {validation.row_count} valid rows; last column “{target}” is the positive "
            "μΩ·cm resistivity target."
        )

    async def _on_model_pick(self, event: rio.FilePickEvent) -> None:
        file = event.file
        if Path(file.name).suffix.lower() != ".joblib":
            self._set_status("The base model must be a .joblib file.", "danger")
            return

        try:
            contents = await file.read_bytes()
            inspection = await asyncio.to_thread(inspect_model_bytes, contents)
        except Exception as error:
            self._set_status(f"Could not read or inspect the model package: {error}", "danger")
            return

        self._model_bytes = contents
        self.model_name = file.name
        self._apply_model_inspection(inspection)
        self.has_results = False
        self.last_summary = {}
        self._fine_tuned_model_bytes = b""
        self._comparison_csv_bytes = b""
        self._metrics_xlsx_bytes = b""
        self._refresh_preflight()

        if self.preflight_ready:
            self._set_status("Model and data preflight passed; fine-tuning is ready.", "success")
        else:
            self._set_status("Base model loaded. " + self.preflight_summary, "info")

    def _apply_model_inspection(self, inspection: ModelInspection) -> None:
        self.model_type = inspection.model_type
        self.model_features = list(inspection.features)
        self.category_mappings = {
            str(feature): {str(label): float(code) for label, code in mapping.items()}
            for feature, mapping in inspection.category_mappings.items()
        }
        self.input_transforms = {
            str(feature): str(transform)
            for feature, transform in inspection.input_transforms.items()
        }
        self.conditional_missing_when_a_zero = list(
            inspection.conditional_missing_when_a_zero
        )

    async def _on_data_pick(self, event: rio.FilePickEvent) -> None:
        file = event.file
        if Path(file.name).suffix.lower() not in {".csv", ".xlsx"}:
            self._set_status("New data must be a CSV or XLSX file.", "danger")
            return

        try:
            contents = await file.read_bytes()
            frame = await asyncio.to_thread(read_dataframe_from_bytes, file.name, contents)
        except Exception as error:
            self._set_status(f"Could not read the new data: {error}", "danger")
            return

        self._data_bytes = contents
        self._data_frame = frame
        self.data_name = file.name
        self.data_row_count = len(frame)
        self.has_results = False
        self.last_summary = {}
        self._fine_tuned_model_bytes = b""
        self._comparison_csv_bytes = b""
        self._metrics_xlsx_bytes = b""
        self._refresh_preflight()

        if self.preflight_ready:
            self._set_status("Model and data preflight passed; fine-tuning is ready.", "success")
        elif self._model_bytes:
            self._set_status(self.preflight_summary, "danger")
        else:
            self._set_status("New data loaded; preflight will run after the base model is uploaded.", "info")

    def _set_num_rounds(self, event: rio.TextInputChangeEvent) -> None:
        self.num_rounds = event.text

    def _set_learning_rate(self, event: rio.TextInputChangeEvent) -> None:
        self.learning_rate = event.text

    @staticmethod
    def _performance_frame(summary: dict[str, Any]) -> pd.DataFrame:
        rows = [
            ("R²", "", summary.get("R2_before"), summary.get("R2_after"), summary.get("delta_R2")),
            (
                "RMSE",
                "log10(μΩ·cm)",
                summary.get("RMSE_before"),
                summary.get("RMSE_after"),
                summary.get("delta_RMSE"),
            ),
            (
                "MAE",
                "log10(μΩ·cm)",
                summary.get("MAE_before"),
                summary.get("MAE_after"),
                summary.get("delta_MAE"),
            ),
            (
                "Tree count",
                "trees",
                summary.get("num_trees_before"),
                summary.get("num_trees_after"),
                (summary.get("num_trees_after") or 0) - (summary.get("num_trees_before") or 0),
            ),
        ]
        return pd.DataFrame(rows, columns=["metric", "unit", "before", "after", "change"])

    @classmethod
    def _fine_tune_in_temporary_directory(
        cls,
        model_bytes: bytes,
        data_bytes: bytes,
        data_name: str,
        rounds: int,
        learning_rate: float,
    ) -> tuple[dict[str, Any], bytes, bytes, bytes]:
        """Run the path-only legacy routine and return browser-ready bytes."""

        from .. import fine_tuning as fine_tuning

        suffix = Path(data_name).suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            raise ValueError("The new-data file extension is invalid")

        with tempfile.TemporaryDirectory(prefix="lsco-finetune-") as directory:
            root = Path(directory)
            model_path = root / "source_model.joblib"
            data_path = root / f"new_data{suffix}"
            output_path = root / "fine_tuned_model.joblib"
            model_path.write_bytes(model_bytes)
            data_path.write_bytes(data_bytes)

            raw_summary = fine_tuning.fine_tune_xgboost_pipeline(
                str(model_path),
                str(data_path),
                str(output_path),
                rounds,
                learning_rate,
                False,
            )
            if not isinstance(raw_summary, dict):
                raise ValueError("The fine-tuning routine did not return a metric summary")

            comparison_path = Path(str(raw_summary.get("comparison_csv") or ""))
            if not output_path.is_file() or not comparison_path.is_file():
                raise ValueError("Fine-tuning did not produce both the model and comparison file")

            summary: dict[str, Any] = {}
            for key in (
                "R2_before",
                "R2_after",
                "RMSE_before",
                "RMSE_after",
                "MAE_before",
                "MAE_after",
                "delta_R2",
                "delta_RMSE",
                "delta_MAE",
            ):
                value = raw_summary.get(key)
                summary[key] = float(value) if value is not None else None
            for key in ("num_trees_before", "num_trees_after"):
                value = raw_summary.get(key)
                summary[key] = int(value) if value is not None else None

            metric_bytes = dataframe_to_xlsx_bytes(cls._performance_frame(summary), "Metrics")
            return (
                summary,
                output_path.read_bytes(),
                comparison_path.read_bytes(),
                metric_bytes,
            )

    async def _run_finetune(self) -> None:
        if self.is_running:
            return
        self._refresh_preflight()
        if not self.preflight_ready:
            self._set_status(self.preflight_summary, "danger")
            return

        try:
            rounds = int(self.num_rounds.strip())
            learning_rate = float(self.learning_rate.strip())
            if rounds < 1 or learning_rate <= 0:
                raise ValueError
        except (TypeError, ValueError):
            self._set_status("Additional tree rounds must be a positive integer and learning rate must be positive.", "danger")
            return

        self.is_running = True
        self._set_status("Fine-tuning the model and calculating Before/After metrics…", "info")
        self.force_refresh()

        try:
            summary, model_bytes, comparison_bytes, metrics_bytes = await asyncio.to_thread(
                self._fine_tune_in_temporary_directory,
                self._model_bytes,
                self._data_bytes,
                self.data_name,
                rounds,
                learning_rate,
            )
        except Exception as error:
            self.is_running = False
            self._set_status(f"Model fine-tuning failed: {error}", "danger")
            return

        self.last_summary = summary
        self._fine_tuned_model_bytes = model_bytes
        self._comparison_csv_bytes = comparison_bytes
        self._metrics_xlsx_bytes = metrics_bytes
        self.has_results = True
        self.is_running = False
        self._set_status("Fine-tuning complete. Save the model, prediction differences, and metrics through the browser.", "success")

    async def _download_model(self) -> None:
        if not self._fine_tuned_model_bytes:
            self._set_status("No fine-tuned model is available for download.", "warning")
            return
        stem = Path(self.model_name).stem or "lsco_model"
        await self.session.save_file(
            self._fine_tuned_model_bytes,
            f"{stem}_fine_tuned.joblib",
            media_type="application/octet-stream",
        )
        self._set_status("The fine-tuned model is ready in the browser download dialog.", "success")

    async def _download_comparison(self) -> None:
        if not self._comparison_csv_bytes:
            self._set_status("No prediction-difference CSV is available for download.", "warning")
            return
        await self.session.save_file(
            self._comparison_csv_bytes,
            "lsco_fine_tune_prediction_comparison.csv",
            media_type="text/csv",
        )
        self._set_status("The prediction-difference CSV is ready in the browser download dialog.", "success")

    async def _download_metrics(self) -> None:
        if not self._metrics_xlsx_bytes:
            self._set_status("No metrics XLSX is available for download.", "warning")
            return
        await self.session.save_file(
            self._metrics_xlsx_bytes,
            "lsco_fine_tune_metrics.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self._set_status("The metrics XLSX is ready in the browser download dialog.", "success")

    def _card(
        self,
        title: str,
        content: rio.Component,
        *,
        step: str,
        icon: str,
        subtitle: str = "",
        fill_height: bool = False,
        expand_content: bool = False,
    ) -> rio.Component:
        """Render one flat, numbered desktop group box."""

        return SectionCard(
            f"{step} · {title}",
            content=content,
            subtitle=subtitle or None,
            icon=icon,
            dense=True,
            fill_height=fill_height,
            expand_content=expand_content,
            grow_x=True,
            grow_y=fill_height,
        )

    @staticmethod
    def _format_metric(value: Any, digits: int = 4) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):.{digits}f}"
        except (TypeError, ValueError):
            return "--"

    @staticmethod
    def _format_delta(value: Any, digits: int = 4) -> str:
        if value is None:
            return "--"
        try:
            return f"{float(value):+.{digits}f}"
        except (TypeError, ValueError):
            return "--"

    def _metric_row(
        self,
        name: str,
        unit: str,
        before: Any,
        after: Any,
        delta: Any,
    ) -> rio.Component:
        digits = 0 if name == "Tree count" else 4
        label = name if not unit else f"{name} · {unit}"
        return rio.Row(
            rio.Text(label, overflow="wrap", grow_x=True),
            rio.Text(
                self._format_metric(before, digits),
                justify="right",
                min_width=5.5,
                overflow="nowrap",
            ),
            rio.Text(
                self._format_metric(after, digits),
                justify="right",
                min_width=5.5,
                overflow="nowrap",
            ),
            rio.Text(
                self._format_delta(delta, digits),
                justify="right",
                min_width=5.5,
                overflow="nowrap",
            ),
            spacing=0.45,
            grow_x=True,
        )

    def _result_component(self) -> rio.Component:
        if not self.last_summary:
            return rio.Row(
                rio.Icon(
                    "material/analytics",
                    fill=rio.Color.from_hex(FIG19_DEEP_BLUE),
                    min_width=2.3,
                    min_height=2.3,
                    align_y=0.5,
                ),
                rio.Column(
                    rio.Text("Waiting for fine-tuning results", style="heading3"),
                    rio.Text("After completion, compare Before, After, and the change here.", style="dim", overflow="wrap"),
                    spacing=0.2,
                    grow_x=True,
                ),
                spacing=0.75,
                grow_x=True,
                grow_y=True,
                align_y=0.5,
            )

        summary = self.last_summary
        header = rio.Rectangle(
            content=rio.Row(
                rio.Text("Metric", font_weight="bold", grow_x=True),
                rio.Text("Before", font_weight="bold", min_width=5.5, justify="right"),
                rio.Text("After", font_weight="bold", min_width=5.5, justify="right"),
                rio.Text("Δ", font_weight="bold", min_width=5.5, justify="right"),
                spacing=0.45,
                grow_x=True,
            ),
            fill=rio.Color.from_hex(FIG19_COOL_PALE),
            corner_radius=0.5,
            margin=0.45,
            grow_x=True,
        )
        return rio.Column(
            rio.Text("Δ = After − Before; RMSE and MAE are measured in log space.", style="dim", overflow="wrap"),
            header,
            self._metric_row(
                "R²",
                "",
                summary.get("R2_before"),
                summary.get("R2_after"),
                summary.get("delta_R2"),
            ),
            rio.Separator(),
            self._metric_row(
                "RMSE",
                "log10(μΩ·cm)",
                summary.get("RMSE_before"),
                summary.get("RMSE_after"),
                summary.get("delta_RMSE"),
            ),
            rio.Separator(),
            self._metric_row(
                "MAE",
                "log10(μΩ·cm)",
                summary.get("MAE_before"),
                summary.get("MAE_after"),
                summary.get("delta_MAE"),
            ),
            rio.Separator(),
            self._metric_row(
                "Tree count",
                "trees",
                summary.get("num_trees_before"),
                summary.get("num_trees_after"),
                (summary.get("num_trees_after") or 0) - (summary.get("num_trees_before") or 0),
            ),
            spacing=0.45,
            grow_x=True,
            grow_y=True,
        )

    def _preflight_component(self) -> rio.Component:
        state_style = "success" if self.preflight_ready else "info"
        if self._model_bytes and self._data_frame is not None and not self.preflight_ready:
            state_style = "warning"
        details: list[rio.Component] = [rio.Text(self.preflight_summary, overflow="wrap")]
        if self.target_name:
            details.append(rio.Text(f"Target column: {self.target_name} (last column, μΩ·cm)", style="dim", overflow="wrap"))
        if self.extra_feature_columns:
            details.append(
                rio.Text(
                    "Extra fields: " + ", ".join(self.extra_feature_columns) + " (not used for training).",
                    style="dim",
                    overflow="wrap",
                )
            )
        return rio.Tooltip(
            anchor=StatusBanner(self.preflight_summary, state_style),
            tip=rio.Column(*details, spacing=0.3, min_width=31, grow_x=True),
            position="right",
            grow_x=True,
        )

    def _download_actions(self) -> rio.Component:
        """Keep every browser download in the same compact action zone."""

        return rio.FlowContainer(
            rio.Button(
                "Download model",
                icon="material/download",
                style="minor",
                is_sensitive=self.has_results and not self.is_running,
                on_press=self._download_model,
            ),
            rio.Button(
                "Difference CSV",
                icon="material/download",
                style="minor",
                is_sensitive=self.has_results and not self.is_running,
                on_press=self._download_comparison,
            ),
            rio.Button(
                "Metrics XLSX",
                icon="material/download",
                style="minor",
                is_sensitive=self.has_results and not self.is_running,
                on_press=self._download_metrics,
            ),
            spacing=0.5,
            justify="grow",
            grow_x=True,
        )

    def _model_upload_content(self) -> rio.Component:
        summary = (
            f"Loaded: {self.model_name} · {self.model_type} · {len(self.model_features)} fields"
            if self.model_name
            else "No model uploaded yet."
        )
        return rio.Column(
            rio.Text("Only trusted joblib model packages are accepted.", style="dim", overflow="wrap"),
            rio.FilePickerArea(
                content="Choose or drop a .joblib model",
                file_types=[".joblib"],
                multiple=False,
                on_pick_file=self._on_model_pick,
                min_height=2.8,
                grow_x=True,
            ),
            rio.Text(summary, overflow="ellipsize"),
            spacing=0.18,
            grow_x=True,
        )

    def _data_upload_content(self) -> rio.Component:
        summary = f"Loaded: {self.data_name} · {self.data_row_count} rows" if self.data_name else "No new data uploaded yet."
        return rio.Column(
            rio.Text("CSV / XLSX; the last column must be a positive μΩ·cm target.", style="dim", overflow="wrap"),
            rio.FilePickerArea(
                content="Choose or drop new experimental data",
                file_types=[".csv", ".xlsx"],
                multiple=False,
                on_pick_file=self._on_data_pick,
                min_height=2.8,
                grow_x=True,
            ),
            rio.Text(summary, overflow="ellipsize"),
            spacing=0.18,
            grow_x=True,
        )

    def _action_content(self) -> rio.Component:
        return rio.Column(
            self._preflight_component(),
            rio.Row(
                rio.TextInput(
                    label="Additional tree rounds",
                    text=self.num_rounds,
                    on_change=self._set_num_rounds,
                    grow_x=True,
                ),
                rio.TextInput(
                    label="Learning rate",
                    text=self.learning_rate,
                    on_change=self._set_learning_rate,
                    grow_x=True,
                ),
                spacing=0.55,
                grow_x=True,
            ),
            rio.Button(
                "Start model fine-tuning",
                icon="material/play_arrow",
                style="major",
                is_loading=self.is_running,
                is_sensitive=self.preflight_ready and not self.is_running,
                on_press=self._run_finetune,
                grow_x=True,
            ),
            rio.Row(
                rio.Icon("material/download", fill=rio.Color.from_hex(FIG19_DEEP_BLUE), align_y=0.5),
                rio.Column(
                    rio.Text("Downloads", font_weight="bold"),
                    rio.Text("Browser-selected location only; temporary files are removed.", style="dim", overflow="ellipsize"),
                    spacing=0.1,
                    grow_x=True,
                ),
                spacing=0.45,
                grow_x=True,
            ),
            self._download_actions(),
            spacing=0.28,
            grow_x=True,
        )

    def build(self) -> rio.Component:
        model_card = self._card(
            "Trusted base model",
            self._model_upload_content(),
            step="01",
            icon="material/psychology",
        )

        data_card = self._card(
            "New experimental data",
            self._data_upload_content(),
            step="02",
            icon="material/dataset",
        )

        controls_card = self._card(
            "Preflight, run, and download",
            self._action_content(),
            step="03",
            icon="material/rocket_launch",
        )

        results_card = self._card(
            "Before → After metrics",
            self._result_component(),
            step="04",
            icon="material/analytics",
            fill_height=True,
            expand_content=True,
        )

        # This laboratory tool is desktop-only.  Keep all four workflow
        # stages visible together; there is no secondary mobile switcher.
        content = ClassContainer(
            content=rio.Row(
                ClassContainer(
                    content=rio.Column(
                        model_card,
                        data_card,
                        controls_card,
                        spacing=0.22,
                        grow_x=True,
                        align_y=0,
                    ),
                    classes=["lsco-parameters-pane"],
                    grow_x=True,
                    grow_y=True,
                ),
                ClassContainer(
                    content=results_card,
                    classes=["lsco-results-pane"],
                    grow_x=True,
                    grow_y=True,
                ),
                spacing=0.42,
                proportions=[0.38, 0.62],
                grow_x=True,
                grow_y=True,
            ),
            classes=["lsco-workbench-split"],
            grow_x=True,
            grow_y=True,
        )

        return PageShell(
            "Model Fine-tuning",
            subtitle="Trusted model + new experimental data → preflight → fine-tuning → local download",
            content=content,
            status_text=self.status_text,
            status_kind=self.status_style,
            model_label=self.model_name or "LSCO model",
            output_unit="μΩ·cm",
            fill_height=True,
            grow_y=True,
        )
