"""Constrained inverse search for candidate LSCO synthesis conditions.

The active GWO-75 model remains a forward regressor.  This page asks the
backend to evaluate a bounded Cartesian candidate grid and ranks the rows by
distance from a positive target resistivity in log space.  It never uploads a
second model and never writes a user-visible result to a server path.
"""

from __future__ import annotations

import asyncio
from dataclasses import field as dataclass_field
from io import BytesIO
import json
import math
from typing import Any, Mapping

import pandas as pd
import requests
import rio
from rio.components.class_container import ClassContainer

from ..components.locale import language_for
from ..components.palette import DESKTOP_BORDER, DESKTOP_SURFACE
from ..components.ui import PageShell, SectionCard
from ..config import BACKEND_URL
from ..workflow_utils import dataframe_to_csv_bytes


DEFAULT_PSYN_MBAR = "1.333223684e-5,0.0001333223684,0.001333223684,0.01333223684,0.2666447368"
DEFAULT_PC_MBAR = DEFAULT_PSYN_MBAR
ALLOWED_SUBSTRATES = ("YSZ", "LAO", "LSAT", "STO", "MgO")
MAX_CANDIDATES = 20_000
PREVIEW_ROWS = 5
PREVIEW_COLUMNS = 4


def _parse_finite_number(text: str, label: str) -> float:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a number.") from exc
    if not math.isfinite(value):
        raise ValueError(f"{label} must be finite.")
    return value


def _parse_float_list(
    text: str,
    label: str,
    *,
    positive: bool = False,
    lower: float | None = None,
    upper: float | None = None,
) -> list[float]:
    tokens = [token.strip() for token in str(text).split(",")]
    if not tokens or any(not token for token in tokens):
        raise ValueError(f"{label} must be a non-empty comma-separated list.")
    values = [_parse_finite_number(token, label) for token in tokens]
    if positive and any(value <= 0.0 for value in values):
        raise ValueError(f"Every {label} value must be greater than 0.")
    if lower is not None and any(value < lower for value in values):
        raise ValueError(f"Every {label} value must be at least {lower:g}.")
    if upper is not None and any(value > upper for value in values):
        raise ValueError(f"Every {label} value must be at most {upper:g}.")
    return values


def _parse_choice_list(text: str, label: str, allowed: tuple[str, ...]) -> list[str]:
    values = [value.strip() for value in str(text).split(",")]
    if not values or any(not value for value in values):
        raise ValueError(f"{label} must be a non-empty comma-separated list.")
    invalid = [value for value in values if value not in allowed]
    if invalid:
        raise ValueError(f"Unknown {label}: {', '.join(invalid)}.")
    return values


def validate_inverse_form(form: Mapping[str, str]) -> tuple[dict[str, Any] | None, dict[str, str], int]:
    """Validate UI text and return a normalized API payload.

    The function is deliberately independent of Rio so malformed ranges and
    candidate-grid limits can be regression-tested without a browser session.
    """

    errors: dict[str, str] = {}

    def number(name: str, label: str, *, positive: bool = False, nonnegative: bool = False) -> float | None:
        try:
            value = _parse_finite_number(form.get(name, ""), label)
            if positive and value <= 0.0:
                raise ValueError(f"{label} must be greater than 0.")
            if nonnegative and value < 0.0:
                raise ValueError(f"{label} must be at least 0.")
            return value
        except ValueError as exc:
            errors[name] = str(exc)
            return None

    def values(
        name: str,
        label: str,
        *,
        positive: bool = False,
        lower: float | None = None,
        upper: float | None = None,
    ) -> list[float] | None:
        try:
            return _parse_float_list(
                form.get(name, ""),
                label,
                positive=positive,
                lower=lower,
                upper=upper,
            )
        except ValueError as exc:
            errors[name] = str(exc)
            return None

    target = number("target_rho", "Target resistivity", positive=True)
    thickness = number("thickness_nm", "Film thickness", positive=True)
    ta = number("ta_deg_c", "Annealing temperature")
    pa = number("pa_mbar", "Annealing oxygen pressure", positive=True)
    ta_h = number("ta_h", "Annealing duration", nonnegative=True)

    try:
        top_n_float = _parse_finite_number(form.get("top_n", ""), "Top candidates")
        top_n = int(top_n_float)
        if top_n_float != top_n or not 1 <= top_n <= 100:
            raise ValueError("Top candidates must be an integer from 1 to 100.")
    except ValueError as exc:
        errors["top_n"] = str(exc)
        top_n = 0

    sr_values = values("sr_values", "Sr", lower=0.0, upper=0.8)
    ts_values = values("ts_values", "Ts")
    tmeas_values = values("tmeas_values", "Tmeas")
    psyn_values = values("psyn_mbar_values", "Psyn", positive=True)
    pc_values = values("pc_mbar_values", "Pc", positive=True)

    try:
        substrates = _parse_choice_list(
            form.get("substrates", ""),
            "substrate",
            ALLOWED_SUBSTRATES,
        )
    except ValueError as exc:
        errors["substrates"] = str(exc)
        substrates = None

    try:
        raw_annealing = _parse_float_list(form.get("annealing_values", ""), "A")
        annealing_values = [int(value) for value in raw_annealing]
        if any(value not in (0.0, 1.0) for value in raw_annealing):
            raise ValueError("A must contain only 0 and/or 1.")
    except ValueError as exc:
        errors["annealing_values"] = str(exc)
        annealing_values = None

    growth_method = str(form.get("growth_method", "")).strip()
    if growth_method not in ("MBE", "PLD"):
        errors["growth_method"] = "Growth method must be MBE or PLD."
    oxygen_activation = str(form.get("oxygen_activation", "")).strip()
    if oxygen_activation not in ("No", "Ozone"):
        errors["oxygen_activation"] = "Oxygen activation must be No or Ozone."

    lists = (
        sr_values,
        ts_values,
        tmeas_values,
        substrates,
        psyn_values,
        pc_values,
        annealing_values,
    )
    candidate_count = math.prod(len(item) for item in lists) if all(item is not None for item in lists) else 0
    if candidate_count > MAX_CANDIDATES:
        errors["candidate_count"] = (
            f"The search grid contains {candidate_count:,} candidates; the limit is {MAX_CANDIDATES:,}."
        )

    if errors:
        return None, errors, candidate_count

    assert None not in (target, thickness, ta, pa, ta_h)
    assert all(item is not None for item in lists)
    payload: dict[str, Any] = {
        "target_rho_uohm_cm": target,
        "top_n": top_n,
        "sr_values": sr_values,
        "ts_values": ts_values,
        "tmeas_values": tmeas_values,
        "substrates": substrates,
        "psyn_mbar_values": psyn_values,
        "pc_mbar_values": pc_values,
        "annealing_values": annealing_values,
        "growth_method": growth_method,
        "oxygen_activation": oxygen_activation,
        "ta_deg_c": ta,
        "pa_mbar": pa,
        "ta_h": ta_h,
        "thickness_nm": thickness,
    }
    return payload, {}, candidate_count


@rio.page(
    name="Inverse Design",
    url_segment="inverse",
    icon="material/hub",
    order=30,
)
class InverseDesignPage(rio.Component):
    """Rank candidate processing windows for a target resistivity."""

    target_rho: str = "500"
    top_n: str = "20"
    growth_method: str = "MBE"
    oxygen_activation: str = "No"
    annealing_values: str = "1"
    ta_deg_c: str = "350"
    pa_mbar: str = "212.275875"
    ta_h: str = "2"
    thickness_nm: str = "30"
    sr_values: str = "0.0675,0.125,0.25,0.50"
    ts_values: str = "600,650,700,750"
    tmeas_values: str = "300"
    substrates: str = "YSZ,LAO,LSAT,STO,MgO"
    psyn_mbar_values: str = DEFAULT_PSYN_MBAR
    pc_mbar_values: str = DEFAULT_PC_MBAR

    model_label: str = "GWO-75"
    target_unit: str = "μΩ·cm"
    metadata_loaded: bool = False
    is_running: bool = False
    has_results: bool = False
    field_errors: dict[str, str] = {}
    candidate_count: int = 0
    top_result_count: int = 0
    best_prediction: str = "—"
    best_error_percent: str = "—"
    extrapolated_features: list[str] = []
    preview_row_page: int = 0
    preview_column_group: int = 0
    preview_notice: str = ""
    status_text_en: str = "Set a positive target and run the bounded candidate search."
    status_text_zh: str = "设置正的目标电阻率，然后运行受限候选搜索。"
    status_kind: str = "info"

    _top_results: pd.DataFrame | None = dataclass_field(default=None, init=False, repr=False)
    _all_results: pd.DataFrame | None = dataclass_field(default=None, init=False, repr=False)
    _preview_frame: pd.DataFrame | None = dataclass_field(default=None, init=False, repr=False)
    _result_csv_bytes: bytes = dataclass_field(default=b"", init=False, repr=False)
    _result_xlsx_bytes: bytes = dataclass_field(default=b"", init=False, repr=False)
    _config_json_bytes: bytes = dataclass_field(default=b"", init=False, repr=False)

    def _tr(self, english: str, chinese: str) -> str:
        return chinese if language_for(self) == "zh" else english

    def _set_status(self, english: str, chinese: str, kind: str = "info") -> None:
        self.status_text_en = english
        self.status_text_zh = chinese
        self.status_kind = kind

    def _form(self) -> dict[str, str]:
        return {
            name: str(getattr(self, name))
            for name in (
                "target_rho",
                "top_n",
                "growth_method",
                "oxygen_activation",
                "annealing_values",
                "ta_deg_c",
                "pa_mbar",
                "ta_h",
                "thickness_nm",
                "sr_values",
                "ts_values",
                "tmeas_values",
                "substrates",
                "psyn_mbar_values",
                "pc_mbar_values",
            )
        }

    @staticmethod
    def _fetch_metadata_sync() -> dict[str, Any]:
        response = requests.get(f"{BACKEND_URL}/features", timeout=15)
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, dict):
            raise ValueError("The backend /features response is not an object.")
        return payload

    @staticmethod
    def _run_search_sync(payload: dict[str, Any]) -> dict[str, Any]:
        response = requests.post(f"{BACKEND_URL}/inverse-design", json=payload, timeout=120)
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict):
            raise ValueError("The inverse-design response is not an object.")
        return result

    @rio.event.on_populate
    async def _load_metadata(self) -> None:
        try:
            payload = await asyncio.to_thread(self._fetch_metadata_sync)
            warning = payload.get("warning")
            if warning:
                raise RuntimeError(str(warning))
            self.model_label = str(payload.get("model_label") or "LSCO model")
            self.target_unit = str(payload.get("target_unit") or "μΩ·cm")
            self.metadata_loaded = True
            self._set_status(
                "GWO-75 is ready. The default grid contains 2,000 candidates.",
                "GWO-75 已就绪。默认搜索网格包含 2,000 个候选。",
                "success",
            )
        except Exception as exc:
            self.metadata_loaded = False
            self._set_status(
                f"Backend unavailable: {exc}",
                f"后端不可用：{exc}",
                "danger",
            )

    def _normalize_results(self, frame: pd.DataFrame) -> pd.DataFrame:
        preferred = [
            "Rank",
            "Predicted_rho_uohm_cm",
            "Target_rho_uohm_cm",
            "Abs_relative_error_percent",
            "Abs_error_log10",
            "Sr",
            "Substrate",
            "Mismatch",
            "Ts",
            "Psyn",
            "Pc",
            "A",
            "TA",
            "PA",
            "tA",
            "t",
            "Tmeas",
            "Oxygen activation",
            "Growth method",
            "Extrapolated_features",
        ]
        columns = [column for column in preferred if column in frame.columns]
        columns.extend(column for column in frame.columns if column not in columns)
        return frame.reindex(columns=columns)

    @staticmethod
    def _xlsx_bytes(
        top_results: pd.DataFrame,
        all_results: pd.DataFrame,
        mismatch_table: pd.DataFrame,
    ) -> bytes:
        stream = BytesIO()
        with pd.ExcelWriter(stream, engine="openpyxl") as writer:
            top_results.to_excel(writer, sheet_name="Top candidates", index=False)
            all_results.to_excel(writer, sheet_name="All candidates", index=False)
            mismatch_table.to_excel(writer, sheet_name="Mismatch table", index=False)
        return stream.getvalue()

    async def _run_search(self) -> None:
        if self.is_running:
            return
        payload, errors, candidate_count = validate_inverse_form(self._form())
        self.field_errors = errors
        self.candidate_count = candidate_count
        if payload is None:
            first_error = next(iter(errors.values()), "Invalid inverse-design input.")
            self._set_status(
                first_error,
                "输入无效：" + first_error,
                "danger",
            )
            return
        if not self.metadata_loaded:
            await self._load_metadata()
            if not self.metadata_loaded:
                return

        self.is_running = True
        self._set_status(
            f"Evaluating {candidate_count:,} candidates…",
            f"正在评估 {candidate_count:,} 个候选……",
            "info",
        )
        try:
            response = await asyncio.to_thread(self._run_search_sync, payload)
            top_rows = response.get("top_candidates")
            all_rows = response.get("all_candidates")
            if not isinstance(top_rows, list) or not isinstance(all_rows, list) or not top_rows:
                raise ValueError("The backend returned no candidate rows.")

            top_results = self._normalize_results(pd.DataFrame(top_rows))
            all_results = self._normalize_results(pd.DataFrame(all_rows))
            mismatch_rows = response.get("mismatch_table", [])
            mismatch_table = pd.DataFrame(mismatch_rows if isinstance(mismatch_rows, list) else [])
            normalized_config = response.get("normalized_config", payload)

            self._top_results = top_results
            self._all_results = all_results
            self._result_csv_bytes = dataframe_to_csv_bytes(all_results)
            self._result_xlsx_bytes = self._xlsx_bytes(top_results, all_results, mismatch_table)
            self._config_json_bytes = json.dumps(
                normalized_config,
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            ).encode("utf-8")
            self.candidate_count = int(response.get("candidate_count", len(all_results)))
            self.top_result_count = len(top_results)
            self.model_label = str(response.get("model_label") or self.model_label)
            self.target_unit = str(response.get("target_unit") or self.target_unit)
            self.extrapolated_features = [
                str(value) for value in response.get("extrapolated_features", [])
            ]
            first = top_results.iloc[0]
            self.best_prediction = f"{float(first['Predicted_rho_uohm_cm']):.6g}"
            self.best_error_percent = f"{float(first['Abs_relative_error_percent']):.4g}%"
            self.has_results = True
            self.preview_row_page = 0
            self.preview_column_group = 0
            self._refresh_preview()

            latency = int(response.get("latency_ms", 0))
            if self.extrapolated_features:
                fields = ", ".join(self.extrapolated_features)
                self._set_status(
                    f"Ranked {self.candidate_count:,} candidates in {latency} ms; extrapolation: {fields}.",
                    f"已在 {latency} ms 内完成 {self.candidate_count:,} 个候选排序；外推字段：{fields}。",
                    "warning",
                )
            else:
                self._set_status(
                    f"Ranked {self.candidate_count:,} candidates in {latency} ms.",
                    f"已在 {latency} ms 内完成 {self.candidate_count:,} 个候选排序。",
                    "success",
                )
        except Exception as exc:
            detail = str(exc)
            response = getattr(exc, "response", None)
            if response is not None:
                try:
                    body = response.json()
                    raw_detail = body.get("detail", body) if isinstance(body, dict) else body
                    if isinstance(raw_detail, dict):
                        detail = str(raw_detail.get("error") or raw_detail)
                    else:
                        detail = str(raw_detail)
                except Exception:
                    pass
            self.has_results = False
            self._set_status(
                f"Inverse search failed: {detail}",
                f"逆向搜索失败：{detail}",
                "danger",
            )
        finally:
            self.is_running = False

    def _preview_page_counts(self) -> tuple[int, int]:
        if self._top_results is None or self._top_results.empty:
            return 1, 1
        return (
            max(1, math.ceil(len(self._top_results) / PREVIEW_ROWS)),
            max(1, math.ceil(len(self._top_results.columns) / PREVIEW_COLUMNS)),
        )

    def _refresh_preview(self) -> None:
        if self._top_results is None or self._top_results.empty:
            self._preview_frame = None
            self.preview_notice = ""
            return
        row_pages, column_pages = self._preview_page_counts()
        self.preview_row_page = min(max(self.preview_row_page, 0), row_pages - 1)
        self.preview_column_group = min(max(self.preview_column_group, 0), column_pages - 1)
        row_start = self.preview_row_page * PREVIEW_ROWS
        column_start = self.preview_column_group * PREVIEW_COLUMNS
        self._preview_frame = self._top_results.iloc[
            row_start : row_start + PREVIEW_ROWS,
            column_start : column_start + PREVIEW_COLUMNS,
        ].copy()
        row_end = min(row_start + PREVIEW_ROWS, len(self._top_results))
        column_end = min(column_start + PREVIEW_COLUMNS, len(self._top_results.columns))
        self.preview_notice = self._tr(
            f"Candidates {row_start + 1}–{row_end} of {len(self._top_results)}; fields {column_start + 1}–{column_end} of {len(self._top_results.columns)}.",
            f"候选 {row_start + 1}–{row_end}/{len(self._top_results)}；字段 {column_start + 1}–{column_end}/{len(self._top_results.columns)}。",
        )

    def _previous_columns(self) -> None:
        self.preview_column_group = max(0, self.preview_column_group - 1)
        self._refresh_preview()

    def _next_columns(self) -> None:
        _, pages = self._preview_page_counts()
        self.preview_column_group = min(pages - 1, self.preview_column_group + 1)
        self._refresh_preview()

    def _previous_rows(self) -> None:
        self.preview_row_page = max(0, self.preview_row_page - 1)
        self._refresh_preview()

    def _next_rows(self) -> None:
        pages, _ = self._preview_page_counts()
        self.preview_row_page = min(pages - 1, self.preview_row_page + 1)
        self._refresh_preview()

    async def _download_csv(self) -> None:
        if self._result_csv_bytes:
            await self.session.save_file(
                self._result_csv_bytes,
                "lsco_inverse_design_results.csv",
                media_type="text/csv",
            )

    async def _download_xlsx(self) -> None:
        if self._result_xlsx_bytes:
            await self.session.save_file(
                self._result_xlsx_bytes,
                "lsco_inverse_design_results.xlsx",
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            )

    async def _download_config(self) -> None:
        if self._config_json_bytes:
            await self.session.save_file(
                self._config_json_bytes,
                "lsco_inverse_design_config.json",
                media_type="application/json",
            )

    def _compact_field(
        self,
        label_en: str,
        label_zh: str,
        state_name: str,
        *,
        help_en: str,
        help_zh: str,
        options: dict[str, str] | None = None,
    ) -> rio.Component:
        error = self.field_errors.get(state_name, "")
        if options is None:
            control: rio.Component = rio.TextInput(
                text=str(getattr(self, state_name)),
                on_change=lambda event, name=state_name: setattr(self, name, event.text),
                is_valid=not bool(error),
                style="rounded",
                grow_x=True,
            )
        else:
            control = rio.Dropdown(
                options=options,
                selected_value=str(getattr(self, state_name)),
                on_change=lambda event, name=state_name: setattr(self, name, str(event.value)),
                is_valid=not bool(error),
                style="rounded",
                grow_x=True,
            )
        label = self._tr(label_en, label_zh)
        help_text = self._tr(help_en, help_zh)
        return rio.Tooltip(
            anchor=rio.Row(
                rio.Text(
                    label,
                    font_weight="bold",
                    overflow="ellipsize",
                    min_width=6.5,
                    align_y=0.5,
                ),
                control,
                spacing=0.28,
                proportions=[0.44, 0.56],
                grow_x=True,
                align_y=0.5,
            ),
            tip=rio.Column(
                rio.Text(help_text, overflow="wrap"),
                *(
                    [rio.Text(error, fill=self.session.theme.danger_color, overflow="wrap")]
                    if error
                    else []
                ),
                spacing=0.18,
                min_width=24,
            ),
            position="right",
            grow_x=True,
        )

    def _field_grid(self, fields: list[rio.Component]) -> rio.Component:
        grid = rio.Grid(row_spacing=0.12, column_spacing=0.45, grow_x=True)
        for index, component in enumerate(fields):
            grid.add(component, row=index // 2, column=index % 2)
        return grid

    @staticmethod
    def _display_value(value: Any) -> str:
        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
        if isinstance(value, float) and math.isfinite(value):
            return f"{value:.6g}"
        text = " ".join(str(value).split())
        return text if len(text) <= 22 else text[:21] + "…"

    def _preview_component(self) -> rio.Component:
        if self._preview_frame is None:
            content: rio.Component = rio.Text(
                self._tr(
                    "Run the search to display ranked candidate conditions.",
                    "运行搜索后将在此显示排序后的候选条件。",
                ),
                style="dim",
                justify="center",
                align_y=0.5,
                grow_x=True,
                grow_y=True,
            )
        else:
            display = self._preview_frame.map(self._display_value)
            rename = {
                "Predicted_rho_uohm_cm": "Pred. ρ (μΩ·cm)",
                "Target_rho_uohm_cm": "Target ρ (μΩ·cm)",
                "Abs_relative_error_percent": "Error (%)",
                "Abs_error_log10": "|Δ log10ρ|",
            }
            display = display.rename(columns=rename)
            table_data = {
                str(column): display[column].astype(str).tolist()
                for column in display.columns
            }
            content = rio.Table(
                data=table_data,
                show_row_numbers=False,
                min_width=0,
                grow_x=True,
            )
        return rio.Rectangle(
            content=content,
            fill=rio.Color.from_hex(DESKTOP_SURFACE),
            stroke_width=0.05,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0.04,
            min_height=12.6,
            grow_x=True,
            grow_y=True,
        )

    def _summary_metric(self, label: str, value: str) -> rio.Component:
        return rio.Rectangle(
            content=rio.Column(
                rio.Text(label, style="dim", overflow="ellipsize", justify="center", grow_x=True),
                rio.Text(value, font_weight="bold", overflow="ellipsize", justify="center", grow_x=True),
                spacing=0.1,
                margin=0.28,
                grow_x=True,
            ),
            fill=rio.Color.from_hex(DESKTOP_SURFACE),
            stroke_width=0.05,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0.04,
            min_width=0,
            grow_x=True,
        )

    def build(self) -> rio.Component:
        fixed_fields = self._field_grid(
            [
                self._compact_field(
                    "Target ρ (μΩ·cm)",
                    "目标 ρ（μΩ·cm）",
                    "target_rho",
                    help_en="Positive target resistivity. Ranking minimizes absolute log10 error.",
                    help_zh="正的目标电阻率；按 log10 绝对误差排序。",
                ),
                self._compact_field(
                    "Top N",
                    "Top N",
                    "top_n",
                    help_en="Number of highest-ranked candidates to preview; 1–100.",
                    help_zh="预览排名最靠前的候选数量，范围 1–100。",
                ),
                self._compact_field(
                    "Growth method",
                    "生长方式",
                    "growth_method",
                    help_en="Categorical model input.",
                    help_zh="模型类别输入。",
                    options={"MBE": "MBE", "PLD": "PLD"},
                ),
                self._compact_field(
                    "O₂ activation",
                    "氧活化方式",
                    "oxygen_activation",
                    help_en="Categorical model input.",
                    help_zh="模型类别输入。",
                    options={"No": "No", "Ozone": "Ozone"},
                ),
                self._compact_field(
                    "Annealing search",
                    "退火状态搜索",
                    "annealing_values",
                    help_en="Search annealed, unannealed, or both cases.",
                    help_zh="搜索退火、未退火或两种状态。",
                    options={"1 = annealed": "1", "0 = not annealed": "0", "0 and 1": "0,1"},
                ),
                self._compact_field(
                    "Film t (nm)",
                    "膜厚 t（nm）",
                    "thickness_nm",
                    help_en="Fixed film thickness for every candidate.",
                    help_zh="全部候选采用的固定膜厚。",
                ),
                self._compact_field(
                    "TA (°C)",
                    "TA（°C）",
                    "ta_deg_c",
                    help_en="Used when A=1; ignored and imputed when A=0.",
                    help_zh="A=1 时使用；A=0 时忽略并由模型填补。",
                ),
                self._compact_field(
                    "PA (mbar)",
                    "PA（mbar）",
                    "pa_mbar",
                    help_en="Positive annealing oxygen pressure used when A=1.",
                    help_zh="A=1 时使用的正退火氧压。",
                ),
                self._compact_field(
                    "tA (h)",
                    "tA（h）",
                    "ta_h",
                    help_en="Non-negative annealing duration used when A=1.",
                    help_zh="A=1 时使用的非负退火时长。",
                ),
            ]
        )
        search_fields = self._field_grid(
            [
                self._compact_field(
                    "Sr values",
                    "Sr 取值",
                    "sr_values",
                    help_en="Comma-separated fractions within 0–0.8.",
                    help_zh="逗号分隔的掺杂分数，范围 0–0.8。",
                ),
                self._compact_field(
                    "Ts values (°C)",
                    "Ts 取值（°C）",
                    "ts_values",
                    help_en="Comma-separated growth temperatures.",
                    help_zh="逗号分隔的生长温度。",
                ),
                self._compact_field(
                    "Tmeas (K)",
                    "Tmeas（K）",
                    "tmeas_values",
                    help_en="Comma-separated measurement temperatures.",
                    help_zh="逗号分隔的测量温度。",
                ),
                self._compact_field(
                    "Substrates",
                    "衬底",
                    "substrates",
                    help_en="YSZ/LAO/LSAT/STO/MgO. Used only to derive Mismatch; not a model feature.",
                    help_zh="可选 YSZ/LAO/LSAT/STO/MgO；仅用于推导 Mismatch，不是模型输入。",
                ),
                self._compact_field(
                    "Psyn (mbar)",
                    "Psyn（mbar）",
                    "psyn_mbar_values",
                    help_en="Comma-separated positive physical pressures; the backend applies log10.",
                    help_zh="逗号分隔的正物理压力；后端自动进行 log10。",
                ),
                self._compact_field(
                    "Pc (mbar)",
                    "Pc（mbar）",
                    "pc_mbar_values",
                    help_en="Comma-separated positive physical pressures; the backend applies log10.",
                    help_zh="逗号分隔的正物理压力；后端自动进行 log10。",
                ),
            ]
        )

        target_card = SectionCard(
            self._tr("Target and fixed conditions", "目标与固定条件"),
            fixed_fields,
            icon="material/tune",
            dense=True,
        )
        search_card = SectionCard(
            self._tr("Discrete search space", "离散搜索空间"),
            search_fields,
            subtitle=self._tr(
                "Comma-separated values; the complete grid is capped at 20,000 candidates.",
                "使用逗号分隔；完整网格最多允许 20,000 个候选。",
            ),
            icon="material/grid_view",
            dense=True,
        )
        action_card = SectionCard(
            self._tr("Run constrained search", "运行受限搜索"),
            rio.Column(
                rio.Text(
                    self._tr(
                        "Substrate derives Mismatch only. Results are ranked candidates, not a unique inverse solution.",
                        "Substrate 仅用于推导 Mismatch。结果是排序候选，并非唯一逆解。",
                    ),
                    style="dim",
                    overflow="ellipsize",
                ),
                rio.Button(
                    self._tr("Start inverse search", "开始逆向搜索"),
                    icon="material/play_arrow",
                    style="major",
                    is_loading=self.is_running,
                    is_sensitive=not self.is_running,
                    on_press=self._run_search,
                ),
                spacing=0.25,
                grow_x=True,
            ),
            icon="material/hub",
            dense=True,
        )

        summary = ClassContainer(
            content=rio.Row(
                self._summary_metric(
                    self._tr("Grid candidates", "网格候选"),
                    f"{self.candidate_count:,}" if self.candidate_count else "—",
                ),
                self._summary_metric(
                    self._tr("Best prediction", "最佳预测"),
                    self.best_prediction,
                ),
                self._summary_metric(
                    self._tr("Relative error", "相对误差"),
                    self.best_error_percent,
                ),
                self._summary_metric(
                    self._tr("Extrapolation", "外推"),
                    str(len(self.extrapolated_features)) if self.has_results else "—",
                ),
                spacing=0.16,
                proportions="homogeneous",
                grow_x=True,
            ),
            classes=["lsco-summary-strip"],
            grow_x=True,
        )
        row_pages, column_pages = self._preview_page_counts()
        controls = rio.Column(
            rio.Row(
                rio.Button(
                    self._tr("Previous fields", "上一组字段"),
                    icon="material/chevron_left",
                    style="minor",
                    is_sensitive=self.has_results and self.preview_column_group > 0,
                    on_press=self._previous_columns,
                ),
                rio.Text(f"{self.preview_column_group + 1}/{column_pages}", align_y=0.5),
                rio.Button(
                    self._tr("Next fields", "下一组字段"),
                    icon="material/chevron_right",
                    style="minor",
                    is_sensitive=self.has_results and self.preview_column_group < column_pages - 1,
                    on_press=self._next_columns,
                ),
                spacing=0.25,
                grow_x=True,
            ),
            rio.Row(
                rio.Button(
                    self._tr("Previous rows", "上一页候选"),
                    icon="material/navigate_before",
                    style="minor",
                    is_sensitive=self.has_results and self.preview_row_page > 0,
                    on_press=self._previous_rows,
                ),
                rio.Text(f"{self.preview_row_page + 1}/{row_pages}", align_y=0.5),
                rio.Button(
                    self._tr("Next rows", "下一页候选"),
                    icon="material/navigate_next",
                    style="minor",
                    is_sensitive=self.has_results and self.preview_row_page < row_pages - 1,
                    on_press=self._next_rows,
                ),
                spacing=0.25,
                grow_x=True,
            ),
            spacing=0.1,
            grow_x=True,
        )
        downloads = rio.Row(
            rio.Button(
                self._tr("Download CSV", "下载 CSV"),
                icon="material/download",
                style="minor",
                is_sensitive=self.has_results and not self.is_running,
                on_press=self._download_csv,
            ),
            rio.Button(
                self._tr("Download XLSX", "下载 XLSX"),
                icon="material/download",
                style="minor",
                is_sensitive=self.has_results and not self.is_running,
                on_press=self._download_xlsx,
            ),
            rio.Button(
                self._tr("Download config", "下载配置"),
                icon="material/data_object",
                style="minor",
                is_sensitive=self.has_results and not self.is_running,
                on_press=self._download_config,
            ),
            spacing=0.35,
            grow_x=True,
        )
        results_card = SectionCard(
            self._tr("Ranked candidate conditions", "候选条件排名"),
            rio.Column(
                summary,
                rio.Text(
                    self.preview_notice
                    or self._tr(
                        "Top candidates are ranked by |Δ log10ρ|; experimental validation is required.",
                        "候选按 |Δ log10ρ| 排名，仍需实验验证。",
                    ),
                    style="dim",
                    overflow="ellipsize",
                ),
                controls,
                self._preview_component(),
                downloads,
                spacing=0.24,
                grow_x=True,
                grow_y=True,
            ),
            icon="material/table_chart",
            dense=True,
            fill_height=True,
            expand_content=True,
        )

        workspace = ClassContainer(
            content=rio.Row(
                ClassContainer(
                    content=rio.Column(
                        target_card,
                        search_card,
                        action_card,
                        spacing=0.28,
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
                proportions=[0.42, 0.58],
                min_width=0,
                grow_x=True,
                grow_y=True,
            ),
            classes=["lsco-workbench-split"],
            grow_x=True,
            grow_y=True,
        )
        return PageShell(
            self._tr("Inverse Design", "逆向设计"),
            subtitle=self._tr(
                "Target resistivity → bounded candidate search → browser downloads",
                "目标电阻率 → 受限候选搜索 → 浏览器下载",
            ),
            content=workspace,
            status_text=(self.status_text_zh if language_for(self) == "zh" else self.status_text_en),
            status_kind=self.status_kind,
            model_label=self.model_label,
            output_unit=self.target_unit,
            fill_height=True,
            grow_y=True,
        )


__all__ = [
    "ALLOWED_SUBSTRATES",
    "DEFAULT_PC_MBAR",
    "DEFAULT_PSYN_MBAR",
    "InverseDesignPage",
    "MAX_CANDIDATES",
    "validate_inverse_form",
]
