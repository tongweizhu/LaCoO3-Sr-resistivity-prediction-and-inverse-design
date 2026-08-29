# zhu/backend_app.py
from __future__ import annotations

import os
import time
import logging
import platform
import math
from typing import List, Dict, Any, Mapping

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:  # Support both ``python -m zhu.backend_app`` and the existing direct import.
    from .inverse_design import (
        INVERSE_DESIGN_NOTE,
        InverseDesignConfig,
        MAX_INVERSE_CANDIDATES,
        build_candidate_grid,
        dataframe_records,
        mismatch_table_records,
        normalized_config as serialize_inverse_config,
        rank_candidates,
    )
except ImportError:  # pragma: no cover - exercised by the deployed direct import
    from inverse_design import (
        INVERSE_DESIGN_NOTE,
        InverseDesignConfig,
        MAX_INVERSE_CANDIDATES,
        build_candidate_grid,
        dataframe_records,
        mismatch_table_records,
        normalized_config as serialize_inverse_config,
        rank_candidates,
    )

# ---------------- Logging ----------------
logger = logging.getLogger("backend")
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# The browser/API contract is always micro-ohm centimetres.  A model package
# may record a log-space offset when its source target was stored in Ω·cm;
# that offset is applied before exposing y_log so this invariant remains true.
RESISTIVITY_UNIT = "μΩ·cm"
LOG_RESISTIVITY_UNIT = "log10(μΩ·cm)"
OUTPUT_TRANSFORM = "y = 10 ** y_log"
INPUT_UNIT_NOTE = "Input feature values must use the displayed physical units and training-data scale; the API applies only the transforms declared by the loaded model and performs no unit conversion."
SUPPORTED_INPUT_TRANSFORMS = {"log10_positive"}

# ---------------- Model Loading ----------------
def _normalise_category_mappings(
    value: object, features: list[str]
) -> dict[str, dict[str, float]]:
    """Validate optional categorical encodings stored in a model package."""

    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("category_mappings 必须是字段到类别编码的字典")

    mappings: dict[str, dict[str, float]] = {}
    for feature, raw_mapping in value.items():
        feature_name = str(feature)
        if feature_name not in features:
            raise ValueError(f"category_mappings 含非模型字段: {feature_name}")
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            raise TypeError(f"{feature_name} 的类别编码必须是非空字典")
        encoded: dict[str, float] = {}
        for label, code in raw_mapping.items():
            code_value = float(code)
            if not math.isfinite(code_value):
                raise ValueError(f"{feature_name} 的类别编码包含非有限数")
            encoded[str(label)] = code_value
        mappings[feature_name] = encoded
    return mappings


def _load_saved_model(model_path: str):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"MODEL_PATH 不存在: {model_path}")

    saved = joblib.load(model_path)
    # ``features`` is accepted for legacy packages. New deployment
    # bundles use ``selected_features`` but retain the old key as an alias.
    for k in ("model", "scaler", "imputer"):
        if k not in saved:
            raise KeyError(f"保存的 joblib 缺少键: '{k}'")
    model = saved["model"]
    scaler = saved["scaler"]
    imputer = saved["imputer"]
    raw_features = saved.get("selected_features", saved.get("features"))
    if not isinstance(raw_features, (list, tuple)) or not raw_features:
        raise TypeError("selected_features 必须是字符串列表")
    feats = list(raw_features)
    if not all(isinstance(x, str) for x in feats):
        raise TypeError("selected_features 必须是字符串列表")
    if len(set(feats)) != len(feats):
        raise ValueError("selected_features 不能含重复字段")

    raw_model_columns = saved.get("model_feature_columns", feats)
    if (
        not isinstance(raw_model_columns, (list, tuple))
        or not raw_model_columns
        or not all(isinstance(x, str) for x in raw_model_columns)
    ):
        raise TypeError("model_feature_columns 必须是与 selected_features 等长的字符串列表")
    model_columns = list(raw_model_columns)
    if (
        len(model_columns) != len(feats)
        or not all(isinstance(x, str) for x in model_columns)
    ):
        raise TypeError("model_feature_columns 必须是与 selected_features 等长的字符串列表")
    if len(set(model_columns)) != len(model_columns):
        raise ValueError("model_feature_columns 不能含重复字段")

    category_mappings = _normalise_category_mappings(saved.get("category_mappings"), list(feats))
    metadata = saved.get("metadata", {})
    if metadata is None:
        metadata = {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata 必须是字典")
    raw_input_transforms = metadata.get(
        "input_transforms", saved.get("input_transforms", {})
    )
    if not isinstance(raw_input_transforms, Mapping):
        raise TypeError("input_transforms 必须是字段到变换名称的字典")
    input_transforms: dict[str, str] = {}
    for raw_feature, raw_transform in raw_input_transforms.items():
        feature = str(raw_feature)
        transform = str(raw_transform)
        if feature not in feats:
            raise ValueError(f"input_transforms 含非模型字段: {feature}")
        if transform not in SUPPORTED_INPUT_TRANSFORMS:
            raise ValueError(f"{feature} 使用不支持的输入变换: {transform}")
        input_transforms[feature] = transform
    try:
        output_log_offset = float(metadata.get("output_log_offset", saved.get("output_log_offset", 0.0)))
    except (TypeError, ValueError) as exc:
        raise TypeError("output_log_offset 必须是有限数字") from exc
    if not math.isfinite(output_log_offset):
        raise ValueError("output_log_offset 必须是有限数字")
    raw_conditional = metadata.get("conditional_missing_when_a_zero", ())
    if not isinstance(raw_conditional, (list, tuple)) or not all(
        isinstance(item, str) for item in raw_conditional
    ):
        raise TypeError("conditional_missing_when_a_zero 必须是字符串列表")
    conditional_missing_when_a_zero = [
        feature for feature in raw_conditional if feature in feats and feature != "A"
    ]
    return (
        saved,
        model,
        scaler,
        imputer,
        list(feats),
        list(model_columns),
        category_mappings,
        input_transforms,
        output_log_offset,
        conditional_missing_when_a_zero,
    )

MODEL_PATH = os.environ.get(
    "MODEL_PATH",
    os.path.join(os.path.dirname(__file__), "models", "LSCO_GWO75_without_lattice_api_v1.joblib"),
)

model_load_warning: str | None = None

try:
    (
        saved,
        model,
        scaler,
        imputer,
        features,
        model_feature_columns,
        category_mappings,
        input_transforms,
        output_log_offset,
        conditional_missing_when_a_zero,
    ) = _load_saved_model(MODEL_PATH)
    logger.info("模型加载成功 | path=%s | n_features=%d", MODEL_PATH, len(features))
except Exception as e:
    logger.exception("模型加载失败: %s", e)
    # 用占位，保证 /healthz 还能跑起来给出错误信息
    saved, model, scaler, imputer, features = None, None, None, None, []
    model_feature_columns, category_mappings, input_transforms, output_log_offset = [], {}, {}, 0.0
    conditional_missing_when_a_zero = []
    # Do not expose the exception here: it can contain an absolute MODEL_PATH.
    # /features remains consumable by the UI even while the model is unavailable.
    model_load_warning = "The model did not load correctly; feature metadata and prediction are unavailable."

# ---------------- FastAPI ----------------
app = FastAPI(title="LSCO Resistivity Predictor")

# 开发阶段放宽 CORS，避免前端联调受阻
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # 如需收紧再改
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------- Schemas ----------------
class PredictReq(BaseModel):
    samples: List[Dict[str, Any]] = Field(..., description="每个样本为 dict：feature -> value")
    debug: bool = Field(False, description="是否返回更详细的诊断信息")

class PredictResp(BaseModel):
    y: List[float] = Field(..., description="预测电阻率，单位为 μΩ·cm")
    y_log: List[float] = Field(..., description="预测 log10(电阻率 / (μΩ·cm))")
    y_unit: str = Field(RESISTIVITY_UNIT, description="y 的单位")
    y_log_unit: str = Field(LOG_RESISTIVITY_UNIT, description="y_log 的单位定义")
    output_transform: str = Field(OUTPUT_TRANSFORM, description="由 y_log 得到 y 的反变换")
    used_features: List[str] = Field(..., description="模型实际使用的特征及其顺序")
    diagnostics: Dict[str, Any] = Field(default_factory=dict, description="输入、输出与预处理诊断信息")


_DEFAULT_INVERSE_CONFIG = InverseDesignConfig()


class InverseDesignReq(BaseModel):
    """A bounded inverse-search request expressed entirely in public units."""

    target_rho_uohm_cm: float = Field(
        _DEFAULT_INVERSE_CONFIG.target_rho_uohm_cm,
        description="Target resistivity in μΩ·cm",
    )
    top_n: int = Field(_DEFAULT_INVERSE_CONFIG.top_n, description="Number of ranked rows in top_candidates")
    sr_values: List[float] = Field(default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.sr_values))
    ts_values: List[float] = Field(default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.ts_values))
    tmeas_values: List[float] = Field(default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.tmeas_values))
    substrates: List[str] = Field(default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.substrates))
    psyn_mbar_values: List[float] = Field(
        default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.psyn_mbar_values)
    )
    pc_mbar_values: List[float] = Field(
        default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.pc_mbar_values)
    )
    annealing_values: List[float] = Field(
        default_factory=lambda: list(_DEFAULT_INVERSE_CONFIG.annealing_values)
    )
    growth_method: str = _DEFAULT_INVERSE_CONFIG.growth_method
    oxygen_activation: str = _DEFAULT_INVERSE_CONFIG.oxygen_activation
    ta_deg_c: float = _DEFAULT_INVERSE_CONFIG.ta_deg_c
    pa_mbar: float = _DEFAULT_INVERSE_CONFIG.pa_mbar
    ta_h: float = _DEFAULT_INVERSE_CONFIG.ta_h
    thickness_nm: float = _DEFAULT_INVERSE_CONFIG.thickness_nm


class InverseDesignResp(BaseModel):
    """Ranked inverse-search candidates; no server-side output paths."""

    top_candidates: List[Dict[str, Any]]
    all_candidates: List[Dict[str, Any]]
    candidate_count: int
    latency_ms: int
    extrapolated_features: List[str]
    extrapolation_counts: Dict[str, int]
    model_label: str
    target_unit: str = RESISTIVITY_UNIT
    mismatch_table: List[Dict[str, float]]
    normalized_config: Dict[str, Any]
    interpretation: str = INVERSE_DESIGN_NOTE

# ---------------- Helpers ----------------
def _raise_debug(where: str, exc: Exception, trace: Dict[str, Any]):
    """
    把任一步骤的错误包装成 HTTP 400，并附带 where/exception/trace
    """
    logger.exception("失败位置=%s | %s", where, exc)
    raise HTTPException(
        status_code=400,
        detail={
            "error": str(exc),
            "where": where,
            "exception": repr(exc),
            "trace": trace,
        },
    )

def _versions() -> Dict[str, str]:
    out = {"python": platform.python_version()}
    try:
        import xgboost as xgb  # type: ignore
        out["xgboost"] = getattr(xgb, "__version__", "?")
    except Exception:
        out["xgboost"] = "not-importable"
    try:
        import sklearn  # type: ignore
        out["sklearn"] = getattr(sklearn, "__version__", "?")
    except Exception:
        out["sklearn"] = "not-importable"
    try:
        out["pandas"] = pd.__version__
    except Exception:
        out["pandas"] = "?"
    return out


def _model_label() -> str:
    """Return a display-safe model name without exposing MODEL_PATH."""
    if isinstance(saved, Mapping):
        metadata = saved.get("metadata")
        if isinstance(metadata, Mapping):
            label = str(metadata.get("model_label") or "").strip()
            if label:
                return os.path.basename(label)[:120]
    return os.path.basename(MODEL_PATH) or "unknown-model"


def _training_ranges() -> Dict[str, Dict[str, float]]:
    """Read the fitted scaler span in the model feature order when available."""
    if scaler is None or not features:
        return {}

    try:
        data_min = list(scaler.data_min_)
        data_max = list(scaler.data_max_)
        if len(data_min) != len(features) or len(data_max) != len(features):
            return {}

        ranges: Dict[str, Dict[str, float]] = {}
        for feature, lower, upper in zip(features, data_min, data_max):
            lower_value = float(lower)
            upper_value = float(upper)
            if input_transforms.get(feature) == "log10_positive":
                lower_value = 10.0 ** lower_value
                upper_value = 10.0 ** upper_value
            if not math.isfinite(lower_value) or not math.isfinite(upper_value):
                return {}
            ranges[feature] = {"min": lower_value, "max": upper_value}
        return ranges
    except Exception:
        # A future preprocessing object might not expose sklearn's fitted
        # MinMaxScaler attributes.  Metadata must remain usable in that case.
        return {}


def _schema_notice() -> str | None:
    """Surface an active legacy-schema mismatch without blocking predictions."""

    legacy_lattice_fields = [field for field in ("a", "c") if field in features]
    if legacy_lattice_fields:
        return (
            "The loaded legacy model still requires "
            + ", ".join(legacy_lattice_fields)
            + ". A model without these fields has not been loaded; do not treat the legacy schema as the new schema."
        )
    return None


def _categorical_feature_metadata() -> dict[str, dict[str, float]]:
    """Return public category labels and their model encodings."""

    return {
        feature: dict(mapping)
        for feature, mapping in category_mappings.items()
        if feature in features
    }


def _encode_category_value(value: object, mapping: Mapping[str, float]) -> float:
    """Map a user-facing category label (or its stored numeric code) to float."""

    if pd.isna(value):
        return float("nan")

    label = str(value).strip()
    if label in mapping:
        return float(mapping[label])

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isfinite(numeric_value) and any(
        math.isclose(numeric_value, encoded) for encoded in mapping.values()
    ):
        return numeric_value
    return float("nan")


def _encode_input_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int], dict[str, int], dict[str, int]]:
    """Encode categorical fields and coerce every model column to numeric.

    The saved imputer continues to own backwards-compatible missing-value
    filling. The returned counters distinguish an unknown category from a
    generic non-numeric numeric-field value so browser clients can block both.
    """

    numeric = pd.DataFrame(index=frame.index)
    invalid_categories: dict[str, int] = {}
    non_numeric: dict[str, int] = {}
    for feature in features:
        raw = frame[feature]
        if feature in category_mappings:
            mapping = category_mappings[feature]
            encoded = raw.map(lambda value: _encode_category_value(value, mapping))
            invalid_mask = raw.notna() & encoded.isna()
            invalid_categories[feature] = int(invalid_mask.sum())
            non_numeric[feature] = 0
            numeric[feature] = encoded.astype(float)
            continue

        encoded = pd.to_numeric(raw, errors="coerce")
        non_numeric[feature] = int((raw.notna() & encoded.isna()).sum())
        invalid_categories[feature] = 0
        numeric[feature] = encoded.astype(float)
    conditional_missing_counts = {
        feature: 0 for feature in conditional_missing_when_a_zero if feature in numeric.columns
    }
    if "A" in numeric.columns and conditional_missing_when_a_zero:
        unannealed = numeric["A"].eq(0.0)
        for feature in conditional_missing_when_a_zero:
            # Feuil5 encodes these values as blanks when a sample has not been
            # annealed. Accept an explicit UI zero for that not-applicable
            # state, then reproduce the saved model's imputer behavior.
            replace_mask = unannealed & numeric[feature].notna()
            conditional_missing_counts[feature] = int(replace_mask.sum())
            numeric.loc[unannealed, feature] = np.nan
    return numeric, invalid_categories, non_numeric, conditional_missing_counts


def _transform_input_frame(
    frame: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Apply saved-model transforms after validating public-unit inputs.

    The public API keeps oxygen pressures in mbar.  GWO-75 was fitted after a
    positive ``log10`` transform, which must occur before its saved imputer and
    scaler. Missing values remain missing for backwards-compatible imputation;
    explicitly supplied zero or negative values are rejected.
    """

    transformed = frame.copy()
    invalid_counts: dict[str, int] = {}
    for feature, transform in input_transforms.items():
        if feature not in transformed.columns:
            continue
        values = pd.to_numeric(transformed[feature], errors="coerce").astype(float)
        if transform == "log10_positive":
            invalid = values.notna() & (values <= 0.0)
            invalid_counts[feature] = int(invalid.sum())
            transformed[feature] = np.log10(values.where(values > 0.0))
    return transformed, invalid_counts

# ---------------- Endpoints ----------------
@app.get("/healthz")
def healthz() -> Dict[str, Any]:
    ok = all(x is not None for x in (saved, model, scaler, imputer)) and len(features) > 0
    return {
        "status": "ok" if ok else "error",
        "model_loaded": ok,
        "model_path": MODEL_PATH,
        "n_features": len(features),
        "features_head": features[:8],
        "versions": _versions(),
        "units": {
            "input": INPUT_UNIT_NOTE,
            "y": RESISTIVITY_UNIT,
            "y_log": LOG_RESISTIVITY_UNIT,
            "output_transform": OUTPUT_TRANSFORM,
            "legacy_multiplier": "No additional 1e6 multiplier is applied to y.",
        },
        "input_transforms": dict(input_transforms),
        "note": "If sklearn/xgboost reports a version warning, prefer the versions used to save the model or migrate it with XGBoost save_model/load_model.",
    }

@app.get("/features")
def get_features() -> Dict[str, Any]:
    """Expose UI metadata while preserving the legacy ``features`` field.

    The response intentionally has the same shape when model loading fails so
    callers can render a useful unavailable state without special parsing.
    """
    model_is_loaded = all(x is not None for x in (saved, model, scaler, imputer)) and bool(features)
    return {
        "features": list(features),
        "temperature_feature": "Tmeas" if "Tmeas" in features else None,
        "training_ranges": _training_ranges(),
        "input_unit_note": INPUT_UNIT_NOTE,
        "target_unit": RESISTIVITY_UNIT,
        "target_log_unit": LOG_RESISTIVITY_UNIT,
        "output_transform": OUTPUT_TRANSFORM,
        "model_label": _model_label(),
        "categorical_features": _categorical_feature_metadata(),
        "input_transforms": dict(input_transforms),
        "feature_kinds": {
            feature: ("categorical" if feature in category_mappings else "numeric")
            for feature in features
        },
        "source_target_unit": (
            saved.get("metadata", {}).get("source_target_unit")
            if isinstance(saved, Mapping) and isinstance(saved.get("metadata"), Mapping)
            else None
        ),
        "output_log_offset": output_log_offset,
        "conditional_missing_when_a_zero": list(conditional_missing_when_a_zero),
        "schema_notice": _schema_notice() if model_is_loaded else None,
        "warning": None if model_is_loaded else (model_load_warning or "The model did not load correctly; prediction is unavailable."),
    }

@app.post("/predict", response_model=PredictResp)
def predict(req: PredictReq) -> PredictResp:
    """
    预测流程（带调试信息）：
    1) 对齐列到训练特征顺序
    2) 编码类别字段并把数值字段转为数字（失败置 NaN）
    3) imputer -> scaler -> model
    4) y = 10**y_log，单位为 μΩ·cm
    """
    if model is None or scaler is None or imputer is None or not features:
        raise HTTPException(status_code=503, detail={"error": "The model is unavailable", "model_path": MODEL_PATH})

    t0 = time.time()
    trace: Dict[str, Any] = {"step": "start", "received_samples": len(req.samples)}

    # --- Step A: 基本校验
    if not isinstance(req.samples, list) or len(req.samples) == 0:
        raise HTTPException(status_code=400, detail={"error": "samples is empty"})

    all_keys = sorted(set(k for s in req.samples for k in s.keys()))
    trace["received_keys"] = all_keys

    # --- Step B: 构造 DataFrame 并对齐列
    try:
        df_raw = pd.DataFrame(req.samples)
        trace["df_raw_shape"] = list(df_raw.shape)
        trace["df_raw_dtypes"] = {c: str(t) for c, t in df_raw.dtypes.items()}
        # 只保留训练时的列，顺序与训练一致
        df = df_raw.reindex(columns=features)
        trace["df_aligned_shape"] = list(df.shape)
    except Exception as e:
        _raise_debug("dataframe_build/reindex", e, trace)

    # 缺失/未知特征统计
    missing_counts = {str(feature): int(count) for feature, count in df.isna().sum().items()}
    unknown = [k for k in all_keys if k not in features]
    trace["missing_counts_before_numeric"] = missing_counts
    trace["unknown_features"] = unknown

    # --- Step C: 编码类别并数值化（保留 NaN，imputer 会处理）
    try:
        (
            df_num,
            invalid_category_counts,
            non_numeric_counts,
            conditional_missing_counts,
        ) = _encode_input_frame(df)
        trace["df_numeric_shape"] = list(df_num.shape)
        trace["nan_counts_after_numeric"] = {
            str(feature): int(count) for feature, count in df_num.isna().sum().items()
        }
    except Exception as e:
        _raise_debug("to_numeric", e, trace)

    # Predictions outside the observed training span are extrapolations.  The
    # saved MinMaxScaler retains those spans, so report them rather than
    # presenting an apparently precise number without context.
    training_ranges: Dict[str, Dict[str, float]] = _training_ranges()
    out_of_training_range_counts: Dict[str, int] = {}
    try:
        for feature, feature_range in training_ranges.items():
            lower_value = feature_range["min"]
            upper_value = feature_range["max"]
            outside = (df_num[feature] < lower_value) | (df_num[feature] > upper_value)
            out_of_training_range_counts[feature] = int(outside.fillna(False).sum())
    except Exception:
        # The model can still predict if a future scaler does not expose these
        # optional fitted attributes; omit the warning metadata in that case.
        training_ranges = {}
        out_of_training_range_counts = {}

    # --- Step D: saved-model input transforms and imputer
    try:
        df_model_numeric, invalid_transform_counts = _transform_input_frame(df_num)
        trace["input_transforms"] = dict(input_transforms)
        trace["invalid_transform_counts"] = invalid_transform_counts
        invalid_transform_features = [
            feature for feature, count in invalid_transform_counts.items() if count > 0
        ]
        if invalid_transform_features:
            raise ValueError(
                "These fields must be positive before log10 preprocessing: "
                + ", ".join(invalid_transform_features)
            )

        # The adopted model was fitted with the original spreadsheet
        # headers. The public API deliberately exposes concise canonical field
        # names, so rename only at this adapter boundary.
        df_for_model = df_model_numeric.copy()
        df_for_model.columns = model_feature_columns
        X_imp = imputer.transform(df_for_model)
        # The saved scaler was fitted with feature names.  Preserve them after
        # imputation to avoid silently bypassing sklearn's alignment checks.
        X_imp_frame = pd.DataFrame(X_imp, columns=model_feature_columns, index=df_num.index)
        trace["X_imp_shape"] = list(getattr(X_imp, "shape", ("?", "?")))
        # 记录样例（前3行）
        try:
            trace["X_imp_head"] = [list(map(float, X_imp[i])) for i in range(min(3, len(X_imp)))]
        except Exception:
            pass
    except Exception as e:
        _raise_debug("imputer.transform", e, trace)

    # --- Step E: scaler
    try:
        scaler_input = X_imp_frame if hasattr(scaler, "feature_names_in_") else X_imp
        X_scaled = scaler.transform(scaler_input)
        trace["X_scaled_shape"] = list(getattr(X_scaled, "shape", ("?", "?")))
    except Exception as e:
        _raise_debug("scaler.transform", e, trace)

    # --- Step F: model.predict
    try:
        model_y_log = [float(v) for v in model.predict(X_scaled)]
        # GWO-75 was trained on log10(ρ / (Ω·cm)). Preserve the
        # long-standing public μΩ·cm API by shifting the log label before
        # applying the existing y = 10 ** y_log contract.
        y_log = [value + output_log_offset for value in model_y_log]
    except Exception as e:
        _raise_debug("model.predict", e, trace)

    # --- Step G: 后处理
    try:
        y = [10.0 ** v for v in y_log]
    except Exception as e:
        _raise_debug("postprocess_y", e, trace)

    latency_ms = int((time.time() - t0) * 1000)

    diagnostics: Dict[str, Any] = {
        "units": {
            "input": INPUT_UNIT_NOTE,
            "y": RESISTIVITY_UNIT,
            "y_log": LOG_RESISTIVITY_UNIT,
            "output_transform": OUTPUT_TRANSFORM,
        },
        "input": {
            "sample_count": int(len(df_num)),
            "received_feature_count": int(len(all_keys)),
            "received_features": all_keys,
            "required_feature_count": int(len(features)),
            "unknown_features": unknown,
            "missing_features_nonzero": [f for f, c in missing_counts.items() if c > 0],
            "missing_counts": missing_counts,
            "non_numeric_features_nonzero": [f for f, c in non_numeric_counts.items() if c > 0],
            "non_numeric_counts": non_numeric_counts,
            "invalid_category_features_nonzero": [
                f for f, c in invalid_category_counts.items() if c > 0
            ],
            "invalid_category_counts": invalid_category_counts,
            "categorical_features": _categorical_feature_metadata(),
            "conditional_missing_when_a_zero": list(conditional_missing_when_a_zero),
            "conditional_missing_counts": conditional_missing_counts,
            "input_transforms": dict(input_transforms),
            "invalid_transform_counts": trace.get("invalid_transform_counts", {}),
            "training_ranges": training_ranges,
            "out_of_training_range_features": [
                f for f, c in out_of_training_range_counts.items() if c > 0
            ],
            "out_of_training_range_counts": out_of_training_range_counts,
            "nan_counts_after_numeric": trace.get("nan_counts_after_numeric", {}),
            "df_raw_dtypes": trace.get("df_raw_dtypes", {}),
            "case_sensitive_feature_note": "Feature names are case-sensitive; categorical values must use the options returned by /features.",
            "first_sample": req.samples[0] if len(req.samples) > 0 else None,
        },
        "output": {
            "sample_count": int(len(y)),
            "y_min": float(min(y)),
            "y_max": float(max(y)),
            "y_log_min": float(min(y_log)),
            "y_log_max": float(max(y_log)),
            "model_y_log_offset": output_log_offset,
            "y_unit": RESISTIVITY_UNIT,
            "y_log_unit": LOG_RESISTIVITY_UNIT,
            "transform": OUTPUT_TRANSFORM,
        },
        "latency_ms": latency_ms,
    }

    # 可选返回更详细的 trace（前端把 debug=true 传进来）
    if req.debug:
        diagnostics["debug_trace"] = trace

    logger.info(
        "predict | n=%d | recv_keys=%d | unknown=%d | latency=%dms",
        len(df_num), len(all_keys), len(unknown), latency_ms
    )

    return PredictResp(
        y=[float(v) for v in y],
        y_log=[float(v) for v in y_log],
        y_unit=RESISTIVITY_UNIT,
        y_log_unit=LOG_RESISTIVITY_UNIT,
        output_transform=OUTPUT_TRANSFORM,
        used_features=features,
        diagnostics=diagnostics,
    )


def _inverse_config_from_request(req: InverseDesignReq) -> InverseDesignConfig:
    """Copy a Pydantic request into the UI-independent immutable config."""

    return InverseDesignConfig(
        target_rho_uohm_cm=req.target_rho_uohm_cm,
        top_n=req.top_n,
        sr_values=tuple(req.sr_values),
        ts_values=tuple(req.ts_values),
        tmeas_values=tuple(req.tmeas_values),
        substrates=tuple(req.substrates),
        psyn_mbar_values=tuple(req.psyn_mbar_values),
        pc_mbar_values=tuple(req.pc_mbar_values),
        annealing_values=tuple(req.annealing_values),
        growth_method=req.growth_method,
        oxygen_activation=req.oxygen_activation,
        ta_deg_c=req.ta_deg_c,
        pa_mbar=req.pa_mbar,
        ta_h=req.ta_h,
        thickness_nm=req.thickness_nm,
    )


@app.post("/inverse-design", response_model=InverseDesignResp)
def run_inverse_design(req: InverseDesignReq) -> InverseDesignResp:
    """Rank a bounded candidate grid with the already-loaded GWO-75 model.

    The request and response use μΩ·cm.  Substrate never reaches the model; it
    is retained only as the human-readable source of the interpolated Mismatch
    feature.  All results remain in memory for browser-side downloads.
    """

    if model is None or scaler is None or imputer is None or not features:
        raise HTTPException(status_code=503, detail={"error": "The model is unavailable"})

    started = time.perf_counter()
    config = _inverse_config_from_request(req)
    try:
        candidates = build_candidate_grid(config)
        normalized = serialize_inverse_config(config)
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={
                "error": str(exc),
                "candidate_limit": MAX_INVERSE_CANDIDATES,
            },
        ) from exc

    # ``predict`` is the single owner of categorical encoding, positive-log
    # pressure transforms, saved imputation/scaling and the source Ω·cm to
    # public μΩ·cm log offset.  Keeping this path shared prevents the inverse
    # endpoint from drifting from the established forward API contract.
    prediction = predict(
        PredictReq(
            samples=candidates.reindex(columns=features).to_dict(orient="records"),
            debug=False,
        )
    )
    try:
        ranked = rank_candidates(
            candidates,
            prediction.y,
            prediction.y_log,
            config.target_rho_uohm_cm,
        )
    except ValueError as exc:  # pragma: no cover - invalid model output guard
        raise HTTPException(
            status_code=500,
            detail={"error": f"The model returned invalid inverse-search predictions: {exc}"},
        ) from exc

    input_diagnostics = prediction.diagnostics.get("input", {})
    raw_counts = input_diagnostics.get("out_of_training_range_counts", {})
    extrapolation_counts = {
        str(feature): int(count) for feature, count in raw_counts.items()
    }
    extrapolated_features = [
        str(feature)
        for feature in input_diagnostics.get("out_of_training_range_features", [])
    ]
    all_records = dataframe_records(ranked)
    top_count = min(int(config.top_n), len(all_records))
    latency_ms = int((time.perf_counter() - started) * 1000)

    return InverseDesignResp(
        top_candidates=all_records[:top_count],
        all_candidates=all_records,
        candidate_count=len(all_records),
        latency_ms=latency_ms,
        extrapolated_features=extrapolated_features,
        extrapolation_counts=extrapolation_counts,
        model_label=_model_label(),
        target_unit=RESISTIVITY_UNIT,
        mismatch_table=mismatch_table_records(),
        normalized_config=normalized,
        interpretation=INVERSE_DESIGN_NOTE,
    )
