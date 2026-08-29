"""Single-sample LSCO resistivity prediction workbench.

The page deliberately validates a complete sample before it calls the legacy
prediction API.  The API still supports imputation for backwards-compatible
programmatic callers, but the scientific UI must never silently replace a
missing experimental value with a training-set mean.
"""

from __future__ import annotations

import asyncio
import io
import math
import time
from datetime import datetime
from typing import Any

import pandas as pd
import requests
import rio
from rio.components.class_container import ClassContainer

from ..components.feature_catalog import (
    get_feature_definition,
    get_feature_hint,
    get_feature_workstation_label,
)
from ..components.palette import (
    DESKTOP_BORDER,
    DESKTOP_INPUT_BORDER,
    DESKTOP_PANEL,
    DESKTOP_SURFACE,
    FIG19_COOL_BLUE,
    FIG19_COOL_PALE,
    FIG19_DEEP_BLUE,
    FIG19_GRAY,
    FIG19_INK,
    FIG19_MUTED_GRAY,
    FIG19_WHITE,
)
from ..components.ui import FeatureField, MetricCard, PageShell, SectionCard
from ..components.locale import language_for, translate
from ..components.validation import validate_required_numeric, validate_temperature_range
from ..config import BACKEND_URL


TEMPERATURE_DEFAULT_MIN = "30"
TEMPERATURE_DEFAULT_MAX = "300"
SAMPLE_COUNT = 60

DEFAULT_TEST_VALUES: dict[str, str] = {
    "Psyn": "1.3E-5",
    "Oxygen activation": "No",
    "Mismatch": "1.96",
    "Ts": "700",
    "A": "1",
    "TA": "350",
    "PA": "212.276",
    "tA": "2",
    "Pc": "1.3E-5",
    "t": "30",
    "Sr": "0.50",
    "Growth method": "MBE",
}

# A curve must not make the workbench reflow after the user presses the main
# action.  Keep its chart slot and its result summary present in both the
# waiting and completed states.  The dimensions are deliberately modest: the
# app is a single-screen control room, while the exported PNG remains the
# full-size publication figure.
RESULT_PLOT_HEIGHT = 13.5


def _format_number(value: object) -> str:
    """Render model ranges and result metrics without visual noise."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    return f"{number:.5g}"


def _format_scientific_tick(value: float, _: int) -> str:
    """Render y-axis ticks as mantissa times power of ten."""
    if value == 0:
        return "0"
    mantissa, exponent = f"{value:.1e}".split("e")
    return rf"$\mathregular{{{mantissa}\times10^{{{int(exponent)}}}}}$"


def _format_scientific_metric(value: object) -> str:
    """Render result-card values with explicit scientific notation."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "--"
    if not math.isfinite(number):
        return "--"
    if number == 0:
        return "0"
    mantissa, exponent = f"{number:.3e}".split("e")
    superscript_digits = str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa}×10{str(int(exponent)).translate(superscript_digits)}"


def validate_prediction_values(
    values: dict[str, str],
    required_features: list[str],
    category_mappings: dict[str, dict[str, float]] | None = None,
    input_transforms: dict[str, str] | None = None,
    conditional_missing_when_a_zero: list[str] | None = None,
) -> dict[str, str]:
    """Return field-specific errors for a complete, finite model sample.

    ``A`` is modeled as a binary feature. Categories supplied by the active
    model are validated as labels; all remaining features are finite numerics.
    Kept pure so the validation rules can be tested without a Rio session.
    """
    return validate_required_numeric(
        values,
        required_features,
        category_mappings=category_mappings,
        input_transforms=input_transforms,
        conditional_missing_when_a_zero=conditional_missing_when_a_zero or (),
    )


def validate_temperature_values(tmin_text: str, tmax_text: str) -> tuple[float | None, float | None, str]:
    """Validate a plotted temperature interval without restricting extrapolation."""
    is_valid, message = validate_temperature_range(tmin_text, tmax_text)
    if not is_valid:
        return None, None, message
    try:
        tmin = float(tmin_text.strip())
        tmax = float(tmax_text.strip())
    except (AttributeError, TypeError, ValueError):
        return None, None, "Enter two temperature values"
    return tmin, tmax, ""


@rio.page(name="Single Prediction", url_segment="", icon="material/show_chart", order=0)
class PredictionPage(rio.Component):
    """Desktop form, curve, diagnostics, and browser-native downloads."""

    metadata: dict[str, Any] = {}
    values: dict[str, str] = {}
    field_errors: dict[str, str] = {}
    tmin: str = TEMPERATURE_DEFAULT_MIN
    tmax: str = TEMPERATURE_DEFAULT_MAX
    temperature_error: str = ""

    status_text: str = "Loading current model metadata…"
    status_kind: str = "info"
    is_loading_metadata: bool = False
    is_predicting: bool = False

    xs: list[float] = []
    ys: list[float] = []
    # A Rio Plot serializes a Matplotlib figure as an SVG with its own natural
    # size.  That natural size can outgrow a fixed workbench slot after a
    # prediction.  Keep a raster preview instead: rio.Image deliberately has
    # no natural size and therefore only occupies the space assigned to it.
    curve_preview_png: bytes | None = None
    y_unit: str = "μΩ·cm"
    last_latency_ms: int = 0
    extrapolated_features: list[str] = []

    @rio.event.on_mount
    async def _on_mount(self) -> None:
        if not self.metadata and not self.is_loading_metadata:
            await self._load_metadata()

    @staticmethod
    def _fetch_metadata() -> dict[str, Any]:
        response = requests.get(f"{BACKEND_URL}/features", timeout=10)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("The backend returned invalid model metadata")
        return data

    async def _load_metadata(self) -> None:
        self.is_loading_metadata = True
        self.status_text = "Loading current model metadata…"
        self.status_kind = "info"
        self.force_refresh()
        try:
            metadata = await asyncio.to_thread(self._fetch_metadata)
            features = metadata.get("features", [])
            temperature_feature = metadata.get("temperature_feature")
            warning = metadata.get("warning")
            schema_notice = str(metadata.get("schema_notice") or "")
            if warning:
                raise RuntimeError(str(warning))
            if not isinstance(features, list) or not features:
                raise RuntimeError("The backend returned no usable model features")
            if temperature_feature != "Tmeas" or temperature_feature not in features:
                raise RuntimeError("The active model has no usable Tmeas feature for a temperature curve")

            self.metadata = metadata
            old_values = self.values
            self.values = {
                feature: (
                    old_values[feature]
                    if feature in old_values
                    else DEFAULT_TEST_VALUES.get(feature, "")
                )
                for feature in features
                if feature != temperature_feature
            }
            self.field_errors = {}
            defaults_are_complete = all(
                str(self.values.get(feature, "")).strip()
                for feature in features
                if feature != temperature_feature
            )
            self.status_text = schema_notice or (
                "Model ready. Default test data loaded; generate the prediction curve."
                if defaults_are_complete
                else "Model ready. Fill every input field, then generate the prediction curve."
            )
            self.status_kind = "warning" if schema_notice else "info"
        except Exception as exc:
            self.metadata = {}
            self.status_text = f"Could not read model metadata: {exc}"
            self.status_kind = "danger"
        finally:
            self.is_loading_metadata = False

    def _features(self) -> list[str]:
        features = self.metadata.get("features", [])
        return [str(feature) for feature in features] if isinstance(features, list) else []

    def _temperature_feature(self) -> str:
        return str(self.metadata.get("temperature_feature", "Tmeas"))

    def _input_features(self) -> list[str]:
        return [feature for feature in self._features() if feature != self._temperature_feature()]

    def _category_mappings(self) -> dict[str, dict[str, float]]:
        raw = self.metadata.get("categorical_features", {})
        if not isinstance(raw, dict):
            return {}
        mappings: dict[str, dict[str, float]] = {}
        for feature, raw_mapping in raw.items():
            if not isinstance(raw_mapping, dict):
                continue
            try:
                mappings[str(feature)] = {
                    str(label): float(code) for label, code in raw_mapping.items()
                }
            except (TypeError, ValueError):
                continue
        return mappings

    def _is_categorical(self, feature: str) -> bool:
        return feature in self._category_mappings()

    def _input_transforms(self) -> dict[str, str]:
        raw = self.metadata.get("input_transforms", {})
        if not isinstance(raw, dict):
            return {}
        return {
            str(feature): str(transform)
            for feature, transform in raw.items()
            if str(feature) in self._features()
        }

    def _training_range(self, feature: str) -> tuple[float | None, float | None]:
        ranges = self.metadata.get("training_ranges", {})
        if not isinstance(ranges, dict):
            return None, None
        item = ranges.get(feature, {})
        if not isinstance(item, dict):
            return None, None
        try:
            lower = float(item["min"])
            upper = float(item["max"])
        except (KeyError, TypeError, ValueError):
            return None, None
        return lower, upper

    def _range_hint(self, feature: str) -> str:
        category_mapping = self._category_mappings().get(feature)
        if category_mapping:
            return "Options: " + " / ".join(category_mapping)
        conditional_fields = self.metadata.get("conditional_missing_when_a_zero", [])
        if self.values.get("A") == "0" and feature in conditional_fields:
            return "Not applicable when unannealed; entering 0 follows the model training convention"
        lower, upper = self._training_range(feature)
        definition = get_feature_definition(feature)
        unit = definition.unit if definition else ""
        if language_for(self) == "en" and unit == "无量纲":
            unit = "dimensionless"
        transform_hint = (
            "enter a positive value; the model applies log10; "
            if self._input_transforms().get(feature) == "log10_positive"
            else ""
        )
        unit_hint = f"Input unit: {unit}; {transform_hint}" if unit else f"Unit pending confirmation; {transform_hint}"
        if lower is None or upper is None:
            return f"{unit_hint}training range unavailable; use the same scale as the training data"
        return f"{unit_hint}training range: {_format_number(lower)} – {_format_number(upper)}"

    def _outside_training_range(self, feature: str, value: float) -> bool:
        lower, upper = self._training_range(feature)
        return lower is not None and upper is not None and (value < lower or value > upper)

    def _set_value(self, feature: str, text: str) -> None:
        updated = dict(self.values)
        updated[feature] = text
        self.values = updated
        if feature in self.field_errors:
            errors = dict(self.field_errors)
            errors.pop(feature, None)
            self.field_errors = errors

    def _set_binary_value(self, feature: str, event: rio.DropdownChangeEvent[str]) -> None:
        self._set_value(feature, str(event.value))
        if feature == "A" and str(event.value) == "0":
            # Feuil5 records the three annealing-only columns as blank for
            # unannealed samples. Keep the visible form complete with zero;
            # the model adapter restores its saved missing-value treatment.
            updated = dict(self.values)
            for annealing_feature in self.metadata.get("conditional_missing_when_a_zero", []):
                if isinstance(annealing_feature, str) and not updated.get(annealing_feature, ""):
                    updated[annealing_feature] = "0"
            self.values = updated

    def _set_tmin(self, event: rio.TextInputChangeEvent) -> None:
        self.tmin = event.text
        self.temperature_error = ""

    def _set_tmax(self, event: rio.TextInputChangeEvent) -> None:
        self.tmax = event.text
        self.temperature_error = ""

    def _preflight_extrapolations(self, tmin: float, tmax: float) -> list[str]:
        outside: list[str] = []
        conditional_fields = self.metadata.get("conditional_missing_when_a_zero", [])
        for feature in self._input_features():
            if self._is_categorical(feature):
                continue
            if self.values.get("A") == "0" and feature in conditional_fields:
                continue
            try:
                value = float(self.values[feature])
            except (KeyError, TypeError, ValueError):
                continue
            if self._outside_training_range(feature, value):
                outside.append(feature)
        if self._outside_training_range(self._temperature_feature(), tmin) or self._outside_training_range(
            self._temperature_feature(), tmax
        ):
            outside.append(self._temperature_feature())
        return outside

    @staticmethod
    def _post_prediction(samples: list[dict[str, Any]]) -> dict[str, Any]:
        response = requests.post(
            f"{BACKEND_URL}/predict",
            json={"samples": samples},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("The backend returned an invalid prediction response")
        return data

    async def _predict(self) -> None:
        if self.is_predicting:
            return
        if not self.metadata:
            await self._load_metadata()
            if not self.metadata:
                return

        input_features = self._input_features()
        category_mappings = self._category_mappings()
        errors = validate_prediction_values(
            self.values,
            input_features,
            category_mappings,
            self._input_transforms(),
            [
                str(feature)
                for feature in self.metadata.get("conditional_missing_when_a_zero", [])
                if isinstance(feature, str)
            ],
        )
        tmin, tmax, temperature_error = validate_temperature_values(self.tmin, self.tmax)
        self.field_errors = errors
        self.temperature_error = temperature_error
        if errors or temperature_error:
            bad_count = len(errors) + bool(temperature_error)
            self.status_text = f"Fix {bad_count} input field(s) before predicting."
            self.status_kind = "danger"
            return
        assert tmin is not None and tmax is not None

        extrapolations = self._preflight_extrapolations(tmin, tmax)
        self.is_predicting = True
        self.status_text = "Generating prediction curve…"
        self.status_kind = "warning" if extrapolations else "info"
        self.force_refresh()

        try:
            temperature_feature = self._temperature_feature()
            xs = [tmin + (tmax - tmin) * index / (SAMPLE_COUNT - 1) for index in range(SAMPLE_COUNT)]
            constant_values: dict[str, Any] = {}
            for feature in input_features:
                constant_values[feature] = (
                    self.values[feature]
                    if feature in category_mappings
                    else float(self.values[feature])
                )
            samples = [
                {**constant_values, temperature_feature: temperature}
                for temperature in xs
            ]

            started = time.perf_counter()
            data = await asyncio.to_thread(self._post_prediction, samples)
            self.last_latency_ms = int((time.perf_counter() - started) * 1000)
            raw_y = data.get("y", [])
            if not isinstance(raw_y, list) or len(raw_y) != len(xs):
                raise RuntimeError("The number of returned prediction points does not match the request")

            self.xs = xs
            self.ys = [float(value) for value in raw_y]
            self.y_unit = str(data.get("y_unit", "μΩ·cm"))
            # Render once at prediction time rather than passing a live
            # Matplotlib Figure to rio.Plot during every page build.  The
            # fixed-size image preview cannot change the layout's natural
            # height when it replaces the empty chart slot.
            self.curve_preview_png = await asyncio.to_thread(self._render_curve_png, 150)
            diagnostics = data.get("diagnostics", {})
            input_diagnostics = diagnostics.get("input", {}) if isinstance(diagnostics, dict) else {}
            backend_extrapolations = input_diagnostics.get("out_of_training_range_features", [])
            all_extrapolations = list(dict.fromkeys([*extrapolations, *backend_extrapolations]))
            self.extrapolated_features = [str(item) for item in all_extrapolations]
            if self.extrapolated_features:
                self.status_text = (
                    f"Generated {len(self.ys)} points; "
                    f"{', '.join(self.extrapolated_features)} are outside the training range; treat as extrapolation."
                )
                self.status_kind = "warning"
            else:
                self.status_text = f"Prediction complete: {len(self.ys)} points in {self.last_latency_ms} ms."
                self.status_kind = "success"
        except Exception as exc:
            self.xs = []
            self.ys = []
            self.curve_preview_png = None
            self.extrapolated_features = []
            self.status_text = f"Prediction failed: {exc}"
            self.status_kind = "danger"
        finally:
            self.is_predicting = False

    def _make_figure(self, dpi: int = 140):
        """Create a light, publication-friendly scatter plot for display/export."""
        # Construct the figure directly on Agg instead of pyplot.  Besides
        # being safe in Rio's worker thread, this keeps the page preview a
        # deterministic in-memory image rather than a GUI-backed Figure.
        from matplotlib import rc_context
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
        from matplotlib.ticker import FuncFormatter, LogLocator

        # Prefer Times New Roman when the host owns it. Tinos is the
        # metric-compatible, redistributable Linux fallback used by the UI.
        with rc_context(
            {
                "font.family": "serif",
                "font.serif": ["Times New Roman", "Tinos", "Liberation Serif", "DejaVu Serif"],
                "font.size": 11,
                "mathtext.fontset": "stix",
                "axes.labelsize": 13,
                "xtick.labelsize": 13,
                "ytick.labelsize": 13,
            }
        ):
            figure = Figure(figsize=(7.2, 4.6), dpi=dpi)
            FigureCanvasAgg(figure)
            axis = figure.add_subplot(111)
            figure.patch.set_facecolor(f"#{FIG19_WHITE}")
            axis.set_facecolor(f"#{FIG19_WHITE}")
            axis.scatter(
                self.xs,
                self.ys,
                s=26,
                marker="o",
                color=f"#{FIG19_DEEP_BLUE}",
                edgecolors=f"#{FIG19_WHITE}",
                linewidths=0.45,
                zorder=3,
            )
            axis.set_xlabel("Tmeas (K)", color=f"#{FIG19_INK}", fontweight="normal", labelpad=5)
            axis.set_ylabel(f"ρ ({self.y_unit})", color=f"#{FIG19_INK}", fontweight="normal", labelpad=12)
            axis.set_yscale("log")
            axis.yaxis.set_minor_locator(LogLocator(base=10.0, subs=tuple(range(2, 10))))
            axis.yaxis.set_major_formatter(FuncFormatter(_format_scientific_tick))
            axis.tick_params(
                axis="both",
                which="major",
                colors=f"#{FIG19_MUTED_GRAY}",
                direction="in",
                top=False,
                right=False,
                length=5,
                width=0.8,
                pad=7,
            )
            axis.tick_params(
                axis="y",
                which="minor",
                colors=f"#{FIG19_MUTED_GRAY}",
                direction="in",
                right=False,
                length=3,
                width=0.6,
            )
            axis.tick_params(axis="x", which="minor", bottom=False, top=False)
            for tick_label in axis.get_xticklabels() + axis.get_yticklabels():
                tick_label.set_fontname("Times New Roman")
            axis.grid(False)
            for spine in ("top", "right", "left", "bottom"):
                axis.spines[spine].set_visible(True)
                axis.spines[spine].set_color(f"#{FIG19_MUTED_GRAY}")
                axis.spines[spine].set_linewidth(0.8)
            figure.subplots_adjust(left=0.18, right=0.98, bottom=0.12, top=0.96)
        return figure

    def _render_curve_png(self, dpi: int = 150) -> bytes:
        """Render a stable in-memory chart preview without imposing SVG size."""

        figure = self._make_figure(dpi=dpi)
        buffer = io.BytesIO()
        try:
            figure.savefig(buffer, format="png", dpi=dpi)
            return buffer.getvalue()
        finally:
            figure.clear()

    async def _download_csv(self) -> None:
        if not self.xs or not self.ys:
            self.status_text = "Generate a prediction curve before downloading CSV."
            self.status_kind = "warning"
            return
        data = pd.DataFrame(
            {
                "Tmeas_K": self.xs,
                "rho_pred_uohm_cm": self.ys,
            }
        ).to_csv(index=False).encode("utf-8")
        filename = f"lsco_resistivity_prediction_{datetime.now():%Y%m%d_%H%M%S}.csv"
        await self.session.save_file(data, filename, media_type="text/csv")
        self.status_text = "CSV is ready in the browser download dialog."
        self.status_kind = "success"

    async def _download_png(self) -> None:
        if not self.xs or not self.ys:
            self.status_text = "Generate a prediction curve before downloading PNG."
            self.status_kind = "warning"
            return

        png = await asyncio.to_thread(self._render_curve_png, 180)
        filename = f"lsco_resistivity_curve_{datetime.now():%Y%m%d_%H%M%S}.png"
        await self.session.save_file(png, filename, media_type="image/png")
        self.status_text = "PNG is ready in the browser download dialog."
        self.status_kind = "success"

    def _feature_input(self, feature: str) -> rio.Component:
        error = self.field_errors.get(feature, "")
        category_mapping = self._category_mappings().get(feature)
        control = FeatureField(
            field=feature,
            label=get_feature_workstation_label(feature, language=language_for(self)),
            unit_hint="",
            text=self.values.get(feature, ""),
            on_change=(
                (lambda event, key=feature: self._set_binary_value(key, event))
                if feature == "A" or category_mapping
                else (lambda event, key=feature: self._set_value(key, event.text))
            ),
            error=error,
            show_error_text=False,
            is_binary=feature == "A",
            binary_options=(
                (
                    {"Select…": "", "0 = not annealed": "0", "1 = annealed": "1"}
                    if language_for(self) == "en"
                    else {"请选择": "", "0 = 未退火": "0", "1 = 已退火": "1"}
                )
                if feature == "A"
                else None
            ),
            choice_options=(
                {
                    ("Select…" if language_for(self) == "en" else "请选择"): "",
                    **{label: label for label in category_mapping},
                }
                if category_mapping
                else None
            ),
        )
        return rio.Tooltip(
            anchor=control,
            tip=rio.Column(
                rio.Text(get_feature_hint(feature, language=language_for(self)), overflow="wrap"),
                rio.Text(translate(self._range_hint(feature), self), style="dim", overflow="wrap"),
                *(
                    [rio.Text(translate(error, self), fill=self.session.theme.danger_color, overflow="wrap")]
                    if error
                    else []
                ),
                spacing=0.18,
                min_width=24,
            ),
            position="right",
            grow_x=True,
        )

    def _input_form(self) -> rio.Component:
        """Render the reference-style single-column desktop parameter form."""

        grid = rio.Grid(row_spacing=0.1, column_spacing=0, grow_x=True)
        for index, feature in enumerate(self._input_features()):
            grid.add(self._feature_input(feature), row=index, column=0)
        return grid

    def _temperature_controls(self) -> rio.Component:
        temperature_feature = self._temperature_feature()
        lower, upper = self._training_range(temperature_feature)
        range_hint = (
            f"Model training temperature range: {_format_number(lower)} – {_format_number(upper)} K."
            if lower is not None and upper is not None
            else "Model training temperature range unavailable."
        )
        controls = rio.Column(
            FeatureField(
                field="Tmin",
                label="Minimum temperature (K)",
                unit_hint="",
                text=self.tmin,
                on_change=self._set_tmin,
            ),
            FeatureField(
                field="Tmax",
                label="Maximum temperature (K)",
                unit_hint="",
                text=self.tmax,
                on_change=self._set_tmax,
            ),
            spacing=0.12,
            grow_x=True,
        )
        scan_note = (
            translate(self.temperature_error, self)
            if self.temperature_error
            else f"Default {TEMPERATURE_DEFAULT_MIN}–{TEMPERATURE_DEFAULT_MAX} K · {range_hint}"
        )
        details: list[rio.Component] = [
            rio.Text(
                scan_note,
                style=(
                    rio.TextStyle(fill=self.session.theme.danger_color)
                    if self.temperature_error
                    else "dim"
                ),
                overflow="ellipsize",
            ),
            controls,
        ]
        return rio.Column(
            rio.Row(
                rio.Icon("material/thermostat", fill=self.session.theme.primary_color, min_width=0.85, min_height=0.85),
                rio.Text("Temperature scan", font_weight="bold", font_size=0.96),
                spacing=0.25,
                align_y=0.5,
            ),
            *details,
            spacing=0.18,
            grow_x=True,
        )

    def _result_panel(self) -> rio.Component:
        """Build a fixed result frame for both waiting and completed states.

        Previously the waiting panel was a short empty-state card and the
        completed panel added a metric grid, a 21-unit plot, and downloads.
        That changed the natural page height after a click, which in turn
        could trigger browser scrollbars and a responsive-layout flip.  This
        method deliberately keeps an identical metric/grid/chart/action
        skeleton throughout the prediction lifecycle.
        """

        has_result = bool(self.xs and self.ys and self.curve_preview_png)
        if has_result:
            applicability = "Extrapolated result" if self.extrapolated_features else "Within model range"
            applicability_detail = (
                f"{len(self.extrapolated_features)} field(s) outside range"
                if self.extrapolated_features
                else "Within saved training range"
            )
            applicability_tip = (
                "Extrapolated fields: " + ", ".join(self.extrapolated_features)
                if self.extrapolated_features
                else "All inputs are within the saved training ranges."
            )
            point_value = f"{len(self.ys)} points"
            point_detail = f"{self.last_latency_ms} ms"
            min_value = _format_scientific_metric(min(self.ys))
            max_value = _format_scientific_metric(max(self.ys))
            chart_surface: rio.Component = rio.Rectangle(
                content=rio.Image(
                    self.curve_preview_png,
                    fill_mode="fit",
                    min_width=0,
                    min_height=RESULT_PLOT_HEIGHT,
                    grow_x=True,
                    grow_y=True,
                ),
                fill=rio.Color.from_hex(FIG19_WHITE),
                stroke_width=0.05,
                stroke_color=rio.Color.from_hex(DESKTOP_INPUT_BORDER),
                corner_radius=0,
                min_height=RESULT_PLOT_HEIGHT,
                grow_x=True,
                grow_y=True,
            )
        else:
            applicability = "Waiting for input"
            applicability_detail = "Range check after generation"
            applicability_tip = "Complete all fields and generate a curve to see the training-range check."
            point_value = "--"
            point_detail = "Waiting for model response"
            min_value = "--"
            max_value = "--"
            chart_surface = rio.Rectangle(
                content=rio.Column(
                    rio.Icon(
                        "material/show_chart",
                        min_width=2.5,
                        min_height=2.5,
                        fill=self.session.theme.secondary_color,
                        align_x=0.5,
                    ),
                    rio.Text(
                        "Complete the inputs to display the prediction curve here.",
                        overflow="ellipsize",
                        align_x=0.5,
                    ),
                    spacing=0.35,
                    align_x=0.5,
                    align_y=0.5,
                    grow_x=True,
                    grow_y=True,
                ),
                fill=rio.Color.from_hex(DESKTOP_SURFACE),
                stroke_width=0.05,
                stroke_color=rio.Color.from_hex(DESKTOP_INPUT_BORDER),
                corner_radius=0,
                min_height=RESULT_PLOT_HEIGHT,
                grow_x=True,
                grow_y=True,
            )

        applicability_card = rio.Tooltip(
            anchor=MetricCard("Model applicability", applicability, applicability_detail),
            tip=rio.Text(applicability_tip, overflow="wrap"),
            position="left",
        )
        metrics = ClassContainer(
            content=rio.Row(
                MetricCard("Prediction series", point_value, point_detail),
                MetricCard("Minimum resistivity", min_value, self.y_unit),
                MetricCard("Maximum resistivity", max_value, self.y_unit),
                applicability_card,
                spacing=0.2,
                proportions="homogeneous",
                grow_x=True,
            ),
            classes=["lsco-summary-strip"],
            grow_x=True,
        )
        chart_slot = ClassContainer(
            content=chart_surface,
            classes=["lsco-plot-region"],
            grow_x=True,
            grow_y=True,
        )
        download_bar = ClassContainer(
            content=rio.Row(
                rio.Button(
                    "Download CSV",
                    icon="material/download",
                    style="minor",
                    on_press=self._download_csv,
                    is_sensitive=has_result,
                ),
                rio.Button(
                    "Download PNG",
                    icon="material/image",
                    style="minor",
                    on_press=self._download_png,
                    is_sensitive=has_result,
                ),
                spacing=0.4,
            ),
            classes=["lsco-download-bar"],
        )
        return SectionCard(
            "Resistivity–Temperature Curve",
            content=rio.Column(
                metrics,
                chart_slot,
                download_bar,
                spacing=0.28,
                grow_x=True,
                grow_y=True,
            ),
            subtitle=f"Current model: {self.metadata.get('model_label', 'Currently loaded model')} · Output unit: {self.y_unit}",
            icon="material/analytics",
            dense=True,
            fill_height=True,
            expand_content=True,
        )

    def build(self) -> rio.Component:
        metadata_note = str(
            self.metadata.get(
                "input_unit_note",
                "Use the same scale as the training data; the page does not convert units automatically.",
            )
        )
        schema_notice = str(self.metadata.get("schema_notice") or "")
        input_content: rio.Component
        if self.is_loading_metadata:
            input_content = rio.Column(
                rio.Text("Loading model features…"),
                rio.ProgressCircle(min_size=3, align_x=0.5),
                spacing=0.6,
                # The surrounding card already has a fixed workbench height,
                # so this placeholder can remain compact without shifting the
                # page or stretching its label and spinner.
                align_x=0.5,
                align_y=0,
            )
        elif not self.metadata:
            input_content = rio.Column(
                rio.Text("Model features are unavailable; prediction is disabled."),
                rio.Button("Reload model metadata", style="minor", icon="material/refresh", on_press=self._load_metadata),
                spacing=0.6,
                align_y=0,
            )
        else:
            input_content = rio.Column(
                rio.Tooltip(
                    anchor=rio.Text("Input scale and range notes", style="dim", overflow="ellipsize"),
                    tip=metadata_note,
                    position="right",
                ),
                *([rio.Banner(text=schema_notice, style="warning")] if schema_notice else []),
                self._input_form(),
                self._temperature_controls(),
                rio.Button(
                    "Generate prediction curve",
                    icon="material/play_arrow",
                    is_loading=self.is_predicting,
                    is_sensitive=not self.is_predicting,
                    on_press=self._predict,
                    grow_x=True,
                ),
                spacing=0.5,
                grow_x=True,
                align_y=0,
            )

        input_panel = SectionCard(
            "Model Inputs",
            content=input_content,
            subtitle="All fields are required; hover a field for its definition and training range.",
            icon="material/tune",
            dense=True,
            fill_height=True,
        )
        result_panel = self._result_panel()
        content: rio.Component = ClassContainer(
            content=rio.Row(
                ClassContainer(
                    content=input_panel,
                    classes=["lsco-parameters-pane"],
                    grow_x=True,
                    grow_y=True,
                ),
                ClassContainer(
                    content=result_panel,
                    classes=["lsco-results-pane"],
                    grow_x=True,
                    grow_y=True,
                ),
                proportions=[0.34, 0.66],
                spacing=0.42,
                grow_x=True,
                grow_y=True,
            ),
            classes=["lsco-workbench-split"],
            grow_x=True,
            grow_y=True,
        )
        return PageShell(
            "LSCO Resistivity Prediction Workbench",
            subtitle="Single-screen inputs · training-range checks · prediction curve · browser downloads",
            content=content,
            status_text=self.status_text,
            status_kind=self.status_kind,
            model_label=str(self.metadata.get("model_label") or "LSCO model"),
            output_unit=self.y_unit,
            fill_height=True,
            # Custom Rio components take their outer growth policy from the
            # component instance rather than from the root returned by
            # ``build``.  This lets PageView actually grant its remaining
            # height to PageShell's fixed-screen workbench.
            grow_y=True,
        )
