"""Browser-first batch prediction workflow for the LSCO predictor.

The page deliberately keeps uploads and generated results in the Rio session.
It never writes a user's spreadsheet or prediction result to a long-lived
server directory; exports are handed to ``session.save_file`` instead.
"""

from __future__ import annotations

import asyncio
import math
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import requests
import rio
from rio.components.class_container import ClassContainer

from ..components.feature_catalog import get_feature_definition, get_feature_label
from ..components.locale import language_for
from ..components.palette import DESKTOP_BORDER, DESKTOP_SURFACE
from ..components.ui import PageShell, SectionCard
from ..config import BACKEND_URL
from ..workflow_utils import (
    REQUIRED_FEATURES,
    FrameValidation,
    count_out_of_range,
    dataframe_to_csv_bytes,
    dataframe_to_xlsx_bytes,
    numeric_feature_frame,
    read_dataframe_from_bytes,
    validate_model_input_frame,
)


@rio.page(
    name="Batch Prediction",
    url_segment="batch",
    icon="material/table_chart",
    order=20,
)
class BatchPredictPage(rio.Component):
    """Validate a spreadsheet, request predictions, and download the result."""

    status_text: str = "Download the template first, or upload a CSV / Excel file containing the model fields."
    status_style: str = "info"
    uploaded_name: str = ""
    row_count: int = 0
    column_count: int = 0

    # Backend metadata is intentionally shown to users so that the page does
    # not silently hard-code a model's applicability range.
    metadata_loaded: bool = False
    model_label: str = ""
    input_unit_note: str = ""
    target_unit: str = "μΩ·cm"
    required_features: list[str] = []
    training_ranges: dict[str, dict[str, float]] = {}
    category_mappings: dict[str, dict[str, float]] = {}
    input_transforms: dict[str, str] = {}
    conditional_missing_when_a_zero: list[str] = []
    schema_notice: str = ""

    missing_columns: list[str] = []
    duplicate_columns: list[str] = []
    extra_columns: list[str] = []
    validation_summary: str = "Column and numeric validation results will appear here after upload."
    range_warning: str = ""

    preview_title: str = "File preview"
    preview_notice: str = ""
    preview_row_page: int = 0
    preview_column_group: int = 0

    is_running: bool = False
    has_results: bool = False
    result_row_count: int = 0
    result_unit: str = "μΩ·cm"

    # These caches are intentionally not Rio state. They can contain large
    # spreadsheets or binaries and should never be serialized to the browser.
    _upload_bytes: bytes = field(default=b"", init=False, repr=False)
    _input_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _preview_source: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _preview_frame: pd.DataFrame | None = field(default=None, init=False, repr=False)
    _result_csv_bytes: bytes = field(default=b"", init=False, repr=False)
    _result_xlsx_bytes: bytes = field(default=b"", init=False, repr=False)

    @staticmethod
    def _fetch_feature_metadata_sync() -> dict[str, Any]:
        response = requests.get(f"{BACKEND_URL}/features", timeout=15)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("The backend /features response is not an object")
        return payload

    async def _ensure_feature_metadata(self, *, announce_error: bool) -> bool:
        """Fetch the current model schema once per browser session."""

        if self.metadata_loaded:
            return True

        try:
            payload = await asyncio.to_thread(self._fetch_feature_metadata_sync)
            warning = payload.get("warning")
            raw_features = payload.get("features")
            if warning:
                raise RuntimeError(str(warning))
            if not isinstance(raw_features, list) or not raw_features:
                raise RuntimeError("The backend did not provide usable model fields")

            features = [str(feature) for feature in raw_features]
            ranges: dict[str, dict[str, float]] = {}
            raw_ranges = payload.get("training_ranges")
            if isinstance(raw_ranges, Mapping):
                for feature, value in raw_ranges.items():
                    if not isinstance(value, Mapping):
                        continue
                    try:
                        ranges[str(feature)] = {
                            "min": float(value["min"]),
                            "max": float(value["max"]),
                        }
                    except (KeyError, TypeError, ValueError):
                        continue

            categories: dict[str, dict[str, float]] = {}
            raw_categories = payload.get("categorical_features")
            if isinstance(raw_categories, Mapping):
                for feature, raw_mapping in raw_categories.items():
                    if not isinstance(raw_mapping, Mapping):
                        continue
                    try:
                        categories[str(feature)] = {
                            str(label): float(code) for label, code in raw_mapping.items()
                        }
                    except (TypeError, ValueError):
                        continue

            self.required_features = features
            self.training_ranges = ranges
            self.category_mappings = categories
            raw_transforms = payload.get("input_transforms", {})
            self.input_transforms = (
                {
                    str(feature): str(transform)
                    for feature, transform in raw_transforms.items()
                    if str(feature) in features
                    and str(transform) == "log10_positive"
                }
                if isinstance(raw_transforms, Mapping)
                else {}
            )
            raw_conditional = payload.get("conditional_missing_when_a_zero", [])
            self.conditional_missing_when_a_zero = [
                str(feature) for feature in raw_conditional
                if isinstance(feature, str) and feature in features
            ] if isinstance(raw_conditional, list) else []
            self.model_label = str(payload.get("model_label") or "Current model")
            self.input_unit_note = str(payload.get("input_unit_note") or "")
            self.target_unit = str(payload.get("target_unit") or "μΩ·cm")
            self.schema_notice = str(payload.get("schema_notice") or "")
            self.metadata_loaded = True
            return True
        except Exception as error:
            if announce_error:
                self._set_status(
                    f"Could not read model metadata; prediction is unavailable: {error}",
                    "danger",
                )
            return False

    @rio.event.on_populate
    async def _load_feature_metadata(self) -> None:
        # Do not replace the welcome message when the backend has not started
        # yet. The predict action will show a precise, actionable error.
        await self._ensure_feature_metadata(announce_error=False)

    def _feature_names(self) -> tuple[str, ...]:
        """Use backend schema whenever available, with a safe display fallback."""

        return tuple(self.required_features) if self.required_features else REQUIRED_FEATURES

    def _set_status(self, text: str, style: str = "info") -> None:
        self.status_text = text
        self.status_style = style

    @staticmethod
    def _validation_message(validation: FrameValidation) -> str:
        messages: list[str] = []
        if validation.missing:
            messages.append("Missing required columns: " + ", ".join(validation.missing))
        if validation.duplicates:
            messages.append("Duplicate columns: " + ", ".join(validation.duplicates))

        invalid_parts: list[str] = []
        for feature, count in validation.empty_cells.items():
            if count:
                invalid_parts.append(f"{feature}: {count} empty")
        for feature, count in validation.non_numeric_cells.items():
            if count:
                invalid_parts.append(f"{feature}: {count} non-numeric")
        for feature, count in validation.non_finite_cells.items():
            if count:
                invalid_parts.append(f"{feature}: {count} non-finite")
        for feature, count in validation.invalid_category_cells.items():
            if count:
                invalid_parts.append(f"{feature}: {count} invalid categories")
        for feature, count in validation.invalid_binary_cells.items():
            if count:
                invalid_parts.append(f"{feature}: {count} values other than 0 or 1")
        for feature, count in validation.non_positive_transform_cells.items():
            if count:
                invalid_parts.append(f"{feature}: {count} values must be greater than 0")
        if invalid_parts:
            messages.append("; ".join(invalid_parts))

        if not messages:
            messages.append("All required columns, numeric values, and categories are valid")
        return ". ".join(messages) + "."

    def _apply_validation(self, validation: FrameValidation) -> None:
        self.missing_columns = list(validation.missing)
        self.duplicate_columns = list(validation.duplicates)
        self.extra_columns = list(validation.extras)
        self.validation_summary = self._validation_message(validation)

    def _current_validation(self) -> FrameValidation | None:
        if self._input_frame is None:
            return None
        return validate_model_input_frame(
            self._input_frame,
            self._feature_names(),
            self.category_mappings,
            input_transforms=self.input_transforms,
            conditional_missing_when_a_zero=self.conditional_missing_when_a_zero,
        )

    def _preview_columns_per_group(self) -> int:
        """Show a stable, desktop-sized slice of the source spreadsheet."""

        return 3

    @staticmethod
    def _preview_rows_per_page() -> int:
        return 4

    def _preview_page_counts(self) -> tuple[int, int]:
        if self._preview_source is None:
            return 1, 1

        row_pages = max(1, math.ceil(len(self._preview_source) / self._preview_rows_per_page()))
        column_pages = max(
            1,
            math.ceil(len(self._preview_source.columns) / self._preview_columns_per_group()),
        )
        return row_pages, column_pages

    def _refresh_preview(self) -> None:
        if self._preview_source is None:
            self._preview_frame = None
            self.preview_notice = ""
            return

        row_pages, column_pages = self._preview_page_counts()
        self.preview_row_page = min(max(self.preview_row_page, 0), row_pages - 1)
        self.preview_column_group = min(
            max(self.preview_column_group, 0), column_pages - 1
        )

        rows_per_page = self._preview_rows_per_page()
        columns_per_group = self._preview_columns_per_group()
        row_start = self.preview_row_page * rows_per_page
        column_start = self.preview_column_group * columns_per_group
        self._preview_frame = self._preview_source.iloc[
            row_start : row_start + rows_per_page,
            column_start : column_start + columns_per_group,
        ].copy()

        if self._preview_source.empty:
            self.preview_notice = "The file was read but contains no data rows."
            return

        row_end = min(row_start + rows_per_page, len(self._preview_source))
        column_end = min(column_start + columns_per_group, len(self._preview_source.columns))
        self.preview_notice = (
            f"Rows {row_start + 1}–{row_end} of {len(self._preview_source)}; "
            f"fields {column_start + 1}–{column_end} of {len(self._preview_source.columns)}."
        )

    def _previous_column_group(self) -> None:
        self.preview_column_group = max(0, self.preview_column_group - 1)
        self._refresh_preview()

    def _next_column_group(self) -> None:
        _, column_pages = self._preview_page_counts()
        self.preview_column_group = min(column_pages - 1, self.preview_column_group + 1)
        self._refresh_preview()

    def _previous_row_page(self) -> None:
        self.preview_row_page = max(0, self.preview_row_page - 1)
        self._refresh_preview()

    def _next_row_page(self) -> None:
        row_pages, _ = self._preview_page_counts()
        self.preview_row_page = min(row_pages - 1, self.preview_row_page + 1)
        self._refresh_preview()

    async def _download_template(self) -> None:
        if not await self._ensure_feature_metadata(announce_error=True):
            return

        template = pd.DataFrame(columns=list(self._feature_names()))
        await self.session.save_file(
            dataframe_to_csv_bytes(template),
            "lsco_batch_template.csv",
            media_type="text/csv",
        )
        self._set_status("The header-only batch template is ready in the browser download dialog.", "success")

    async def _on_file_pick(self, event: rio.FilePickEvent) -> None:
        file = event.file
        filename = file.name
        suffix = Path(filename).suffix.lower()
        if suffix not in {".csv", ".xlsx"}:
            self._set_status("Only CSV and XLSX files are supported.", "danger")
            return

        try:
            contents = await file.read_bytes()
            frame = await asyncio.to_thread(read_dataframe_from_bytes, filename, contents)
        except Exception as error:
            self._set_status(f"Could not read the uploaded file: {error}", "danger")
            return

        self._upload_bytes = contents
        self._input_frame = frame
        self._preview_source = frame
        self.uploaded_name = filename
        self.row_count = len(frame)
        self.column_count = len(frame.columns)
        self.preview_title = "Input file preview"
        self.preview_row_page = 0
        self.preview_column_group = 0
        self.has_results = False
        self.result_row_count = 0
        self._result_csv_bytes = b""
        self._result_xlsx_bytes = b""
        self.range_warning = ""
        self._refresh_preview()

        metadata_ok = await self._ensure_feature_metadata(announce_error=False)
        validation = self._current_validation()
        assert validation is not None
        self._apply_validation(validation)

        if not metadata_ok:
            self._set_status(
                "The file is loaded, but model metadata is unavailable. The preview is available and the backend will be checked again before prediction.",
                "warning",
            )
        elif not validation.is_valid:
            self._set_status("The file is loaded, but validation failed. Correct it and upload again.", "danger")
        elif validation.extras:
            self._set_status(
                "Validation passed. Extra columns will remain in the export but will not be used for prediction.",
                "warning",
            )
        else:
            self._set_status("Validation passed; batch prediction is ready to run.", "success")

    @staticmethod
    def _post_predict_sync(samples: list[dict[str, float]]) -> dict[str, Any]:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"samples": samples},
            timeout=120,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("The backend prediction response is not an object")
        return payload

    async def _run_predict(self) -> None:
        if self.is_running:
            return
        if self._input_frame is None:
            self._set_status("Upload and read a data file first.", "danger")
            return
        if not await self._ensure_feature_metadata(announce_error=True):
            return

        validation = self._current_validation()
        assert validation is not None
        self._apply_validation(validation)
        if not validation.is_valid:
            self._set_status("Data validation failed: " + self.validation_summary, "danger")
            return

        try:
            numeric_frame = numeric_feature_frame(
                self._input_frame,
                self._feature_names(),
                self.category_mappings,
            )
        except Exception as error:
            self._set_status(f"Could not convert model inputs to numeric values: {error}", "danger")
            return

        range_frame = numeric_frame.copy()
        if "A" in range_frame.columns and self.conditional_missing_when_a_zero:
            unannealed = range_frame["A"].eq(0.0)
            for feature in self.conditional_missing_when_a_zero:
                range_frame.loc[unannealed, feature] = float("nan")
        out_of_range = count_out_of_range(range_frame, self.training_ranges)
        self.range_warning = ""
        if out_of_range:
            summary = "; ".join(f"{name}: {count} row(s)" for name, count in out_of_range.items())
            self.range_warning = f"Outside the training range (extrapolation): {summary}."

        samples = [
            {feature: float(value) for feature, value in row.items()}
            for row in numeric_frame.to_dict(orient="records")
        ]
        self.is_running = True
        self._set_status(f"Predicting {len(samples)} row(s)…", "info")
        self.force_refresh()

        try:
            payload = await asyncio.to_thread(self._post_predict_sync, samples)
            raw_predictions = payload.get("y")
            if not isinstance(raw_predictions, list) or len(raw_predictions) != len(self._input_frame):
                raise ValueError("The backend returned a different number of predictions than input rows")
            predictions = [float(value) for value in raw_predictions]
            if not all(math.isfinite(value) for value in predictions):
                raise ValueError("The backend returned non-finite predictions")

            result = self._input_frame.copy()
            result["rho_pred_uohm_cm"] = predictions
            csv_bytes, xlsx_bytes = await asyncio.gather(
                asyncio.to_thread(dataframe_to_csv_bytes, result),
                asyncio.to_thread(dataframe_to_xlsx_bytes, result, "Predictions"),
            )
        except Exception as error:
            self._set_status(f"Batch prediction failed: {error}", "danger")
            self.is_running = False
            return

        self._result_csv_bytes = csv_bytes
        self._result_xlsx_bytes = xlsx_bytes
        self.has_results = True
        self.result_row_count = len(result)
        self.result_unit = str(payload.get("y_unit") or self.target_unit or "μΩ·cm")
        self._preview_source = result
        self.preview_title = "Result preview (original columns + rho_pred_uohm_cm)"
        self.preview_row_page = 0
        self.preview_column_group = 0
        self._refresh_preview()
        self.is_running = False

        if self.range_warning:
            self._set_status(
                f"Prediction complete for {len(result)} row(s). {self.range_warning} Interpret extrapolated results with care.",
                "warning",
            )
        else:
            self._set_status(
                f"Prediction complete for {len(result)} row(s); rho_pred_uohm_cm is in {self.result_unit}.",
                "success",
            )

    async def _download_result_csv(self) -> None:
        if not self._result_csv_bytes:
            self._set_status("No CSV result is available; run prediction first.", "warning")
            return
        stem = Path(self.uploaded_name).stem or "lsco_batch"
        await self.session.save_file(
            self._result_csv_bytes,
            f"{stem}_predictions.csv",
            media_type="text/csv",
        )
        self._set_status("The CSV result is ready in the browser download dialog.", "success")

    async def _download_result_xlsx(self) -> None:
        if not self._result_xlsx_bytes:
            self._set_status("No XLSX result is available; run prediction first.", "warning")
            return
        stem = Path(self.uploaded_name).stem or "lsco_batch"
        await self.session.save_file(
            self._result_xlsx_bytes,
            f"{stem}_predictions.xlsx",
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        self._set_status("The XLSX result is ready in the browser download dialog.", "success")

    def _card(self, title: str, content: rio.Component, *, icon: str) -> rio.Component:
        return SectionCard(
            title,
            content=content,
            icon=icon,
            dense=True,
            grow_x=True,
        )

    def _field_label_text(self) -> str:
        labels: list[str] = []
        for feature in self._feature_names():
            definition = get_feature_definition(feature)
            if definition is None:
                labels.append(str(feature))
            else:
                labels.append(get_feature_label(feature, language=language_for(self)))
        return (", " if language_for(self) == "en" else "、").join(labels)

    @staticmethod
    def _compact_preview_text(value: Any, *, limit: int = 18) -> str:
        """Format values for an at-a-glance table rather than a spreadsheet."""

        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass

        if isinstance(value, float) and math.isfinite(value):
            text = f"{value:.6g}"
        else:
            text = " ".join(str(value).split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    @staticmethod
    def _compact_preview_header(value: Any, *, limit: int = 16) -> str:
        text = " ".join(str(value).split())
        return text if len(text) <= limit else text[: limit - 1] + "…"

    def _display_preview_frame(self) -> pd.DataFrame:
        """Return a deliberately small, collision-safe presentation frame."""

        assert self._preview_frame is not None
        display = self._preview_frame.copy()
        used_headers: dict[str, int] = {}
        headers: list[str] = []
        for raw_header in display.columns:
            base = self._compact_preview_header(raw_header)
            occurrence = used_headers.get(base, 0) + 1
            used_headers[base] = occurrence
            headers.append(base if occurrence == 1 else f"{base} {occurrence}")
        display.columns = headers
        return display.map(self._compact_preview_text)

    def _preview_component(self) -> rio.Component:
        if self._preview_frame is None:
            return rio.Text("Upload a file to see a compact four-row preview.", overflow="wrap")

        row_pages, column_pages = self._preview_page_counts()
        table = rio.Table(
            data=self._display_preview_frame(),
            show_row_numbers=True,
            min_width=0,
            grow_x=True,
        )
        elements: list[rio.Component] = [
            rio.Text(
                "Compact preview: four rows and a small field set at a time; use the controls instead of scrolling.",
                style="dim",
                overflow="wrap",
            ),
            rio.FlowContainer(
                rio.Button(
                    "Previous fields",
                    icon="material/chevron_left",
                    style="minor",
                    is_sensitive=self.preview_column_group > 0,
                    on_press=self._previous_column_group,
                ),
                rio.Text(
                    f"Field group {self.preview_column_group + 1}/{column_pages}",
                    align_y=0.5,
                ),
                rio.Button(
                    "Next fields",
                    icon="material/chevron_right",
                    style="minor",
                    is_sensitive=self.preview_column_group < column_pages - 1,
                    on_press=self._next_column_group,
                ),
                rio.Button(
                    "Previous rows",
                    icon="material/navigate_before",
                    style="minor",
                    is_sensitive=self.preview_row_page > 0,
                    on_press=self._previous_row_page,
                ),
                rio.Text(
                    f"Row page {self.preview_row_page + 1}/{row_pages}",
                    align_y=0.5,
                ),
                rio.Button(
                    "Next rows",
                    icon="material/navigate_next",
                    style="minor",
                    is_sensitive=self.preview_row_page < row_pages - 1,
                    on_press=self._next_row_page,
                ),
                spacing=0.6,
                justify="grow",
                grow_x=True,
            ),
        ]
        if self.preview_notice:
            elements.append(rio.Text(self.preview_notice, style="dim", overflow="wrap"))
        elements.append(table)
        return rio.Column(*elements, spacing=0.8, grow_x=True)

    def _validation_component(self) -> rio.Component:
        details: list[rio.Component] = [rio.Text(self.validation_summary, overflow="wrap")]
        if self.missing_columns:
            details.append(rio.Text("Missing columns: " + ", ".join(self.missing_columns), overflow="wrap"))
        if self.duplicate_columns:
            details.append(rio.Text("Duplicate columns: " + ", ".join(self.duplicate_columns), overflow="wrap"))
        if self.extra_columns:
            details.append(
                rio.Banner(
                    text="Extra columns: " + ", ".join(self.extra_columns) + ". They remain in the export but are not used for prediction.",
                    style="info",
                )
            )
        if self.range_warning:
            details.append(rio.Banner(text=self.range_warning, style="warning"))
        visible_summary = self.validation_summary
        if self.extra_columns:
            visible_summary += f" Extra columns retained: {len(self.extra_columns)}."
        if self.range_warning:
            visible_summary += " Extrapolation warning available."
        return rio.Tooltip(
            anchor=rio.Text(visible_summary, overflow="ellipsize", grow_x=True),
            tip=rio.Column(*details, spacing=0.35, min_width=32, grow_x=True),
            position="right",
            grow_x=True,
        )

    def build(self) -> rio.Component:
        metadata_text = (
            f"Current model: {self.model_label}; {len(self._feature_names())} model fields required."
            if self.metadata_loaded
            else "Model metadata loads when the backend is available and is rechecked on upload and prediction."
        )
        field_text = self._field_label_text()
        field_summary = f"Model fields: {len(self._feature_names())} · hover for the complete list"
        unit_note = self.input_unit_note or "Use the same scale as the training data; input units are not converted automatically."

        upload_card = self._card(
            "1. Download template and upload data",
            rio.Column(
                rio.Text("Required columns may be in any order; CSV and XLSX are supported.", overflow="wrap"),
                rio.Row(
                    rio.Button(
                        "CSV template",
                        icon="material/download",
                        style="minor",
                        on_press=self._download_template,
                    ),
                    rio.Text("Headers only; use the training-data units and scale.", overflow="wrap", grow_x=True),
                    spacing=0.5,
                    proportions=[0.48, 0.52],
                    grow_x=True,
                    align_y=0.5,
                ),
                rio.Text(
                    "Unannealed rows: set A, TA, PA, and tA to 0.",
                    style="dim",
                    overflow="wrap",
                ),
                rio.FilePickerArea(
                    content="Upload CSV / XLSX data",
                    file_types=[".csv", ".xlsx"],
                    multiple=False,
                    on_pick_file=self._on_file_pick,
                    min_height=3.3,
                    grow_x=True,
                ),
                rio.Text(
                    f"Loaded: {self.uploaded_name} ({self.row_count} rows, {self.column_count} columns)"
                    if self.uploaded_name
                    else "No file uploaded yet.",
                    overflow="ellipsize",
                ),
                spacing=0.38,
                grow_x=True,
            ),
            icon="material/upload_file",
        )

        schema_card = self._card(
            "2. Model fields and validation",
            rio.Column(
                rio.Text(metadata_text, overflow="wrap"),
                rio.Tooltip(
                    anchor=rio.Text(field_summary, style="dim", overflow="ellipsize"),
                    tip=rio.Column(
                        rio.Text("Model fields (headers must use variable names)", font_weight="bold"),
                        rio.Text(field_text, overflow="wrap"),
                        spacing=0.2,
                        min_width=27,
                    ),
                    position="right",
                ),
                rio.Tooltip(
                    anchor=rio.Text("Input units and scale", style="dim", overflow="ellipsize"),
                    tip=rio.Text(unit_note, overflow="wrap"),
                    position="right",
                ),
                *(
                    [rio.Banner(text=self.schema_notice, style="warning")]
                    if self.schema_notice
                    else []
                ),
                self._validation_component(),
                spacing=0.38,
                grow_x=True,
            ),
            icon="material/fact_check",
        )

        action_card = self._card(
            "3. Run and download results",
            rio.Column(
                rio.Text("The result adds rho_pred_uohm_cm (μΩ·cm) after the original columns.", overflow="wrap"),
                rio.Button(
                    "Run batch prediction",
                    icon="material/play_arrow",
                    style="major",
                    is_loading=self.is_running,
                    is_sensitive=not self.is_running and bool(self.uploaded_name),
                    on_press=self._run_predict,
                ),
                rio.FlowContainer(
                    rio.Button(
                    "Download CSV",
                        icon="material/download",
                        style="minor",
                        is_sensitive=self.has_results and not self.is_running,
                        on_press=self._download_result_csv,
                    ),
                    rio.Button(
                    "Download XLSX",
                        icon="material/download",
                        style="minor",
                        is_sensitive=self.has_results and not self.is_running,
                        on_press=self._download_result_xlsx,
                    ),
                    spacing=1,
                    justify="grow",
                    grow_x=True,
                ),
                rio.Text(
                    f"Generated {self.result_row_count} result rows." if self.has_results else "After the run, both result formats can be saved in the browser.",
                    overflow="ellipsize",
                ),
                spacing=0.38,
                grow_x=True,
            ),
            icon="material/play_circle",
        )

        def summary_metric(label: str, value: str) -> rio.Component:
            return rio.Rectangle(
                content=rio.Column(
                    rio.Text(label, style="dim", overflow="ellipsize", justify="center", grow_x=True),
                    rio.Text(value, overflow="ellipsize", justify="center", grow_x=True),
                    spacing=0.12,
                    margin=0.3,
                    grow_x=True,
                    grow_y=True,
                ),
                fill=rio.Color.from_hex(DESKTOP_SURFACE),
                stroke_width=0.05,
                stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
                corner_radius=0.04,
                min_width=0,
                grow_x=True,
                grow_y=True,
            )

        summary_strip = ClassContainer(
            content=rio.Row(
                summary_metric("Input rows", str(self.row_count) if self.uploaded_name else "—"),
                summary_metric("Columns", str(self.column_count) if self.uploaded_name else "—"),
                summary_metric("Extra columns", str(len(self.extra_columns)) if self.uploaded_name else "—"),
                summary_metric("Result rows", str(self.result_row_count) if self.has_results else "—"),
                spacing=0.16,
                proportions="homogeneous",
                grow_x=True,
                grow_y=True,
            ),
            classes=["lsco-summary-strip"],
            grow_x=True,
        )
        preview_card = SectionCard(
            "Result preview" if self.has_results else self.preview_title,
            content=rio.Column(
                summary_strip,
                self._preview_component(),
                spacing=0.3,
                grow_x=True,
                grow_y=True,
            ),
            icon="material/table_chart",
            dense=True,
            fill_height=True,
            expand_content=True,
        )

        # This laboratory tool is desktop-only.  Keep every workflow stage
        # visible at once instead of introducing a second, switcher-based
        # layout at narrower window sizes.
        workspace_content = ClassContainer(
            content=rio.Row(
                ClassContainer(
                    content=rio.Column(
                        upload_card,
                        schema_card,
                        action_card,
                        spacing=0.35,
                        grow_x=True,
                        align_y=0,
                    ),
                    classes=["lsco-parameters-pane"],
                    grow_x=True,
                    grow_y=True,
                ),
                ClassContainer(
                    content=preview_card,
                    classes=["lsco-results-pane"],
                    grow_x=True,
                    grow_y=True,
                ),
                spacing=0.42,
                proportions=[0.34, 0.66],
                min_width=0,
                grow_x=True,
                grow_y=True,
            ),
            classes=["lsco-workbench-split"],
            grow_x=True,
            grow_y=True,
        )

        return PageShell(
            "Batch Prediction",
            subtitle="Template → upload → validate → predict → browser downloads",
            content=workspace_content,
            status_text=self.status_text,
            status_kind=self.status_style,
            model_label=self.model_label or "LSCO model",
            output_unit=self.result_unit if self.has_results else self.target_unit,
            fill_height=True,
            grow_y=True,
        )
