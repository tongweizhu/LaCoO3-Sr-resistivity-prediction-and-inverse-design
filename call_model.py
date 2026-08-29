# -*- coding: utf-8 -*-
"""Run the current GWO-75 LSCO resistivity model on complete example samples.

The default bundle is the API adapter for the supplied GWO-75 Feuil5 model. It
uses 13 fields (including two categorical fields) and deliberately excludes
the historical lattice constants ``a`` and ``c``.  The underlying model stores
``log10(ρ / (Ω·cm))``; its adapter adds the recorded log-space offset before
this script exposes ``y_log`` and ``y`` in ``μΩ·cm``.

Set ``MODEL_PATH`` only to another API-compatible bundle that declares an
``output_log_offset``. Set ``OUTPUT_PATH`` to a ``.csv`` or ``.xlsx`` path
only when a result file should be written.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd


RESISTIVITY_UNIT = "μΩ·cm"
LOG_RESISTIVITY_UNIT = "log10(μΩ·cm)"
WORKSPACE_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = (
    WORKSPACE_DIR / "zhu" / "models" / "LSCO_GWO75_without_lattice_api_v1.joblib"
)


def configured_path(variable_name: str, default: Path) -> Path:
    """Resolve an optional environment-path override relative to the CWD."""
    path = Path(os.environ.get(variable_name, str(default))).expanduser()
    return path.resolve()


def save_results(results: pd.DataFrame) -> None:
    """Optionally write results when OUTPUT_PATH is explicitly configured."""
    output_value = os.environ.get("OUTPUT_PATH")
    if not output_value:
        print("No result file written. Set OUTPUT_PATH to a .csv or .xlsx path to save one.")
        return

    output_path = configured_path("OUTPUT_PATH", WORKSPACE_DIR / "Predicted_new_samples.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".csv":
        results.to_csv(output_path, index=False)
    elif suffix == ".xlsx":
        results.to_excel(output_path, index=False)
    else:
        raise ValueError("OUTPUT_PATH must end in .csv or .xlsx")
    print(f"Results saved to: {output_path}")


def _public_category_mappings(
    raw_mappings: object,
    public_features: list[str],
    model_columns: list[str],
) -> dict[str, dict[str, float]]:
    """Normalise categorical mappings to the public feature names."""
    if raw_mappings is None:
        return {}
    if not isinstance(raw_mappings, Mapping):
        raise TypeError("category_mappings must be a mapping")

    mappings: dict[str, dict[str, float]] = {}
    for raw_feature, raw_mapping in raw_mappings.items():
        feature = str(raw_feature)
        if feature not in public_features:
            if feature not in model_columns:
                raise ValueError(f"Unknown categorical feature in model: {feature}")
            feature = public_features[model_columns.index(feature)]
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            raise TypeError(f"Category mapping for {feature} must be non-empty")
        mappings[feature] = {str(label): float(code) for label, code in raw_mapping.items()}
    return mappings


def _encode_samples(
    frame: pd.DataFrame,
    features: list[str],
    category_mappings: Mapping[str, Mapping[str, float]],
    conditional_missing_when_a_zero: list[str],
) -> pd.DataFrame:
    """Validate and encode public input fields without silently imputing them."""
    missing_columns = [feature for feature in features if feature not in frame.columns]
    if missing_columns:
        raise ValueError("Missing required features: " + ", ".join(missing_columns))

    numeric = pd.DataFrame(index=frame.index)
    for feature in features:
        raw = frame[feature]
        if feature in category_mappings:
            mapping = category_mappings[feature]

            def encode_category(value: object) -> float:
                if pd.isna(value):
                    return float("nan")
                label = str(value).strip()
                if label in mapping:
                    return float(mapping[label])
                try:
                    code = float(value)
                except (TypeError, ValueError):
                    return float("nan")
                return code if any(math.isclose(code, value) for value in mapping.values()) else float("nan")

            encoded = raw.map(encode_category).astype(float)
            invalid = raw.notna() & encoded.isna()
            if invalid.any():
                choices = ", ".join(category_mappings[feature])
                raise ValueError(f"Invalid values for {feature}; use one of: {choices}")
            numeric[feature] = encoded
        else:
            numeric[feature] = pd.to_numeric(raw, errors="coerce")

    non_finite = ~np.isfinite(numeric.to_numpy(dtype=float, copy=False))
    if non_finite.any():
        locations = [
            f"row {row}, {feature}"
            for row, feature in zip(*np.where(non_finite))
        ]
        raise ValueError("Every input must be a finite number or a valid category: " + ", ".join(locations))

    if "A" in numeric.columns and not numeric["A"].isin((0.0, 1.0)).all():
        raise ValueError("A must be 0 (not annealed) or 1 (annealed)")

    # In Feuil5 the three annealing fields are blank whenever A=0. The public
    # UI presents zero for this not-applicable state; restore NaN before the
    # saved imputer runs so the model receives its original training semantics.
    if "A" in numeric.columns:
        unannealed = numeric["A"].eq(0.0)
        for feature in conditional_missing_when_a_zero:
            if feature in numeric:
                numeric.loc[unannealed, feature] = np.nan
    return numeric


def _apply_input_transforms(
    frame: pd.DataFrame,
    input_transforms: Mapping[str, str],
) -> pd.DataFrame:
    """Convert public-unit values to the fitted model's preprocessing space."""

    transformed = frame.copy()
    for feature, transform in input_transforms.items():
        if feature not in transformed.columns:
            raise ValueError(f"Input transform refers to an unknown feature: {feature}")
        if transform != "log10_positive":
            raise ValueError(f"Unsupported input transform for {feature}: {transform}")
        invalid = transformed[feature].notna() & (transformed[feature] <= 0.0)
        if invalid.any():
            raise ValueError(f"{feature} must be positive before log10 preprocessing")
        transformed[feature] = np.log10(
            transformed[feature].where(transformed[feature] > 0.0)
        )
    return transformed


# Step 1. Load the current API-compatible GWO-75 bundle.
model_path = configured_path("MODEL_PATH", DEFAULT_MODEL_PATH)
if not model_path.is_file():
    raise FileNotFoundError(
        f"Saved model not found: {model_path}. Set MODEL_PATH to a valid API adapter .joblib package."
    )

saved = joblib.load(model_path)
for required_key in ("model", "scaler", "imputer"):
    if required_key not in saved:
        raise KeyError(f"Saved joblib package is missing required key: {required_key}")

model = saved["model"]
scaler = saved["scaler"]
imputer = saved["imputer"]
raw_features = saved.get("selected_features", saved.get("features"))
if not isinstance(raw_features, (list, tuple)) or not raw_features:
    raise KeyError("Saved joblib package must provide selected_features or features")
features = [str(feature) for feature in raw_features]
model_columns = [str(column) for column in saved.get("model_feature_columns", features)]
if len(model_columns) != len(features):
    raise ValueError("model_feature_columns must have the same length as selected_features")

metadata = saved.get("metadata", {})
if not isinstance(metadata, Mapping):
    raise TypeError("metadata must be a mapping")
raw_offset = metadata.get("output_log_offset", saved.get("output_log_offset"))
if raw_offset is None:
    raise KeyError(
        "This model does not declare output_log_offset, so its displayed unit cannot be verified. "
        "Use the current GWO-75 API adapter instead."
    )
output_log_offset = float(raw_offset)
if not math.isfinite(output_log_offset):
    raise ValueError("output_log_offset must be finite")

category_mappings = _public_category_mappings(
    saved.get("category_mappings"), features, model_columns
)
raw_conditional = metadata.get("conditional_missing_when_a_zero", ())
if not isinstance(raw_conditional, (list, tuple)):
    raise TypeError("conditional_missing_when_a_zero must be a list")
conditional_missing_when_a_zero = [
    str(feature) for feature in raw_conditional if str(feature) in features
]
raw_input_transforms = metadata.get("input_transforms", saved.get("input_transforms", {}))
if not isinstance(raw_input_transforms, Mapping):
    raise TypeError("input_transforms must be a mapping")
input_transforms = {
    str(feature): str(transform) for feature, transform in raw_input_transforms.items()
}

print(f"Loaded GWO-75 API adapter: {model_path}")
print("Expected features:", features)
print(f"Output: y = 10 ** y_log in {RESISTIVITY_UNIT} ({LOG_RESISTIVITY_UNIT} before conversion)")


# Step 2. Prepare complete samples using the public, no-a/c feature schema.
# For A=0, zeros mark TA/PA/tA as not applicable and are restored to the
# source model's blank-value handling before preprocessing.
samples: list[dict[str, Any]] = [
    {
        "Psyn": 1.33322e-5,
        "Oxygen activation": "No",
        "Mismatch": 1.96,
        "Ts": 700,
        "A": 1,
        "TA": 350,
        "PA": 212.276,
        "tA": 2,
        "Pc": 1.33322e-5,
        "t": 30,
        "Sr": 0.50,
        "Tmeas": 30,
        "Growth method": "MBE",
    },
    {
        "Psyn": 1.33322e-5,
        "Oxygen activation": "No",
        "Mismatch": 1.96,
        "Ts": 700,
        "A": 1,
        "TA": 350,
        "PA": 212.276,
        "tA": 2,
        "Pc": 1.33322e-5,
        "t": 30,
        "Sr": 0.50,
        "Tmeas": 300,
        "Growth method": "MBE",
    },
]

df_input = pd.DataFrame(samples).reindex(columns=features)
df_numeric = _encode_samples(
    df_input, features, category_mappings, conditional_missing_when_a_zero
)

if hasattr(scaler, "data_min_") and hasattr(scaler, "data_max_"):
    out_of_range = [
        feature
        for feature, lower, upper in zip(features, scaler.data_min_, scaler.data_max_)
        if (
            (
                df_numeric[feature]
                < (
                    10.0 ** float(lower)
                    if input_transforms.get(feature) == "log10_positive"
                    else float(lower)
                )
            )
            | (
                df_numeric[feature]
                > (
                    10.0 ** float(upper)
                    if input_transforms.get(feature) == "log10_positive"
                    else float(upper)
                )
            )
        ).fillna(False).any()
    ]
    if out_of_range:
        print("Warning: values outside the model training range (extrapolation):", out_of_range)
print(f"Input samples shape: {df_numeric.shape}")


# Step 3. Reapply the saved preprocessing pipeline and its public-unit adapter.
model_input = _apply_input_transforms(df_numeric, input_transforms).rename(
    columns=dict(zip(features, model_columns))
)
X_imp = imputer.transform(model_input)
X_imp_frame = pd.DataFrame(X_imp, columns=model_columns, index=model_input.index)
X_scaled = scaler.transform(X_imp_frame if hasattr(scaler, "feature_names_in_") else X_imp)
y_pred_log_model = np.asarray(model.predict(X_scaled), dtype=float)
y_pred_log = y_pred_log_model + output_log_offset
y_pred_rho = np.power(10.0, y_pred_log)

results = df_input.copy()
results["Predicted_log10_rho_(log10(μΩ·cm))"] = y_pred_log
results["Predicted_rho_(μΩ·cm)"] = y_pred_rho

print("\nPrediction results:")
print(results[["Predicted_log10_rho_(log10(μΩ·cm))", "Predicted_rho_(μΩ·cm)"]])

save_results(results)
