"""Small, UI-independent helpers for the LSCO workflow pages.

The browser pages deliberately keep uploaded files in memory.  These helpers
make the validation rules testable without requiring a running Rio session or
writing a user's input to a long-lived server directory.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd


# Fallback for the bundled GWO-75 Feuil5 model. Browser pages still fetch the
# active schema from ``GET /features`` so a deliberately supplied legacy model
# remains usable through MODEL_PATH.
REQUIRED_FEATURES: tuple[str, ...] = (
    "Psyn",
    "Oxygen activation",
    "Mismatch",
    "Ts",
    "A",
    "TA",
    "PA",
    "tA",
    "Pc",
    "t",
    "Sr",
    "Tmeas",
    "Growth method",
)

# ``A`` is a confirmed binary model field, not an arbitrary continuous
# numerical input. Keep this rule separate from the runtime feature list so a
# legacy model can still be rendered while the known binary semantics remain
# enforced wherever the field is present.
BINARY_FEATURES: tuple[str, ...] = ("A",)


@dataclass(frozen=True)
class FrameValidation:
    """Validation result for a frame used as model input."""

    missing: tuple[str, ...] = ()
    extras: tuple[str, ...] = ()
    duplicates: tuple[str, ...] = ()
    empty_cells: Mapping[str, int] = field(default_factory=dict)
    non_numeric_cells: Mapping[str, int] = field(default_factory=dict)
    non_finite_cells: Mapping[str, int] = field(default_factory=dict)
    invalid_category_cells: Mapping[str, int] = field(default_factory=dict)
    invalid_binary_cells: Mapping[str, int] = field(default_factory=dict)
    non_positive_transform_cells: Mapping[str, int] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not (
            self.missing
            or self.duplicates
            or any(self.empty_cells.values())
            or any(self.non_numeric_cells.values())
            or any(self.non_finite_cells.values())
            or any(self.invalid_category_cells.values())
            or any(self.invalid_binary_cells.values())
            or any(self.non_positive_transform_cells.values())
        )

    @property
    def invalid_cell_count(self) -> int:
        return sum(self.empty_cells.values()) + sum(
            self.non_numeric_cells.values()
        ) + sum(self.non_finite_cells.values()) + sum(self.invalid_category_cells.values()) + sum(
            self.invalid_binary_cells.values()
        ) + sum(self.non_positive_transform_cells.values())


@dataclass(frozen=True)
class FineTuneValidation:
    """Preflight result for a fine-tuning dataset."""

    frame: FrameValidation
    target_name: str = ""
    target_empty_cells: int = 0
    target_non_numeric_cells: int = 0
    target_non_positive_cells: int = 0
    row_count: int = 0

    @property
    def is_valid(self) -> bool:
        return (
            self.frame.is_valid
            and self.row_count >= 2
            and self.target_empty_cells == 0
            and self.target_non_numeric_cells == 0
            and self.target_non_positive_cells == 0
        )


@dataclass(frozen=True)
class ModelInspection:
    """Safe-to-display metadata read from a user-selected model package."""

    features: tuple[str, ...]
    model_type: str
    category_mappings: Mapping[str, Mapping[str, float]] = field(default_factory=dict)
    input_transforms: Mapping[str, str] = field(default_factory=dict)
    conditional_missing_when_a_zero: tuple[str, ...] = ()


def read_dataframe_from_bytes(name: str, contents: bytes) -> pd.DataFrame:
    """Read a CSV/XLS/XLSX upload without creating a persistent upload file."""

    suffix = Path(name).suffix.lower()
    source = BytesIO(contents)
    if suffix == ".csv":
        frame = pd.read_csv(source)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(source)
    else:
        raise ValueError("Only CSV, XLS, and XLSX files are supported")

    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The file is empty or is not a valid table")

    # Names copied from spreadsheets frequently contain invisible surrounding
    # whitespace.  Normalize only that benign variation; do not guess aliases.
    frame = frame.rename(columns=lambda value: str(value).strip())
    return frame


def dataframe_to_csv_bytes(frame: pd.DataFrame) -> bytes:
    """Create a UTF-8-with-BOM CSV suitable for spreadsheet users."""

    return frame.to_csv(index=False).encode("utf-8-sig")


def dataframe_to_xlsx_bytes(frame: pd.DataFrame, sheet_name: str = "Results") -> bytes:
    """Create an XLSX download entirely in memory."""

    stream = BytesIO()
    with pd.ExcelWriter(stream, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name=sheet_name[:31] or "Results")
    return stream.getvalue()


def _duplicate_names(columns: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for column in columns:
        if column in seen and column not in duplicates:
            duplicates.append(column)
        seen.add(column)
    return tuple(duplicates)


def _category_mapping_for(
    feature: str,
    category_mappings: Mapping[str, Mapping[str, Any]] | None,
) -> Mapping[str, Any] | None:
    if not category_mappings:
        return None
    mapping = category_mappings.get(feature)
    return mapping if isinstance(mapping, Mapping) and mapping else None


def encode_categorical_series(series: pd.Series, mapping: Mapping[str, Any]) -> pd.Series:
    """Encode category labels while also accepting an existing numeric code."""

    allowed_codes = {float(value) for value in mapping.values()}

    def encode(value: object) -> float:
        if pd.isna(value):
            return float("nan")
        label = str(value).strip()
        if label in mapping:
            return float(mapping[label])
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return float("nan")
        return numeric if np.isfinite(numeric) and numeric in allowed_codes else float("nan")

    return series.map(encode).astype(float)


def validate_model_input_frame(
    frame: pd.DataFrame,
    required_features: Sequence[str] = REQUIRED_FEATURES,
    category_mappings: Mapping[str, Mapping[str, Any]] | None = None,
    binary_features: Sequence[str] = BINARY_FEATURES,
    input_transforms: Mapping[str, str] | None = None,
    conditional_missing_when_a_zero: Sequence[str] = (),
) -> FrameValidation:
    """Require complete, finite numeric/category/binary values for every feature.

    The prediction API intentionally supports imputation for backwards
    compatibility.  The new workflow UI is stricter so an uploaded batch is
    auditable and does not silently rely on imputed values.
    """

    required = tuple(required_features)
    columns = tuple(str(column) for column in frame.columns)
    duplicates = _duplicate_names(columns)
    missing = tuple(feature for feature in required if feature not in columns)
    extras = tuple(column for column in columns if column not in required)

    empty_cells: dict[str, int] = {}
    non_numeric_cells: dict[str, int] = {}
    non_finite_cells: dict[str, int] = {}
    invalid_category_cells: dict[str, int] = {}
    invalid_binary_cells: dict[str, int] = {}
    non_positive_transform_cells: dict[str, int] = {}
    known_binary_fields = {str(feature) for feature in binary_features}
    transforms = input_transforms or {}
    conditional = {str(feature) for feature in conditional_missing_when_a_zero}
    unannealed_mask = pd.Series(False, index=frame.index)
    if "A" in columns and "A" not in duplicates:
        a_series = frame["A"]
        if isinstance(a_series, pd.Series):
            unannealed_mask = pd.to_numeric(a_series, errors="coerce").eq(0.0)

    for feature in required:
        # A duplicate required column is invalid already; do not attempt to
        # reduce a multi-column DataFrame to a Series.
        if feature not in columns or feature in duplicates:
            continue

        series = frame[feature]
        if not isinstance(series, pd.Series):
            continue
        text_empty = series.map(lambda value: isinstance(value, str) and not value.strip())
        empty_mask = series.isna() | text_empty
        category_mapping = _category_mapping_for(feature, category_mappings)
        if category_mapping is not None:
            encoded = encode_categorical_series(series, category_mapping)
            invalid_category_mask = (~empty_mask) & encoded.isna()
            empty_cells[feature] = int(empty_mask.sum())
            non_numeric_cells[feature] = 0
            non_finite_cells[feature] = 0
            invalid_category_cells[feature] = int(invalid_category_mask.sum())
            invalid_binary_cells[feature] = 0
            non_positive_transform_cells[feature] = 0
            continue

        numeric = pd.to_numeric(series, errors="coerce")
        non_numeric_mask = (~empty_mask) & numeric.isna()
        non_finite_mask = numeric.notna() & ~np.isfinite(numeric.astype(float))
        binary_mask = (
            numeric.notna()
            & np.isfinite(numeric.astype(float))
            & ~numeric.astype(float).isin((0.0, 1.0))
            if feature in known_binary_fields
            else pd.Series(False, index=series.index)
        )
        non_positive_transform_mask = (
            numeric.notna()
            & np.isfinite(numeric.astype(float))
            & (numeric.astype(float) <= 0.0)
            if transforms.get(feature) == "log10_positive"
            else pd.Series(False, index=series.index)
        )
        if feature in conditional:
            non_positive_transform_mask &= ~unannealed_mask

        empty_cells[feature] = int(empty_mask.sum())
        non_numeric_cells[feature] = int(non_numeric_mask.sum())
        non_finite_cells[feature] = int(non_finite_mask.sum())
        invalid_category_cells[feature] = 0
        invalid_binary_cells[feature] = int(binary_mask.sum())
        non_positive_transform_cells[feature] = int(non_positive_transform_mask.sum())

    return FrameValidation(
        missing=missing,
        extras=extras,
        duplicates=duplicates,
        empty_cells=empty_cells,
        non_numeric_cells=non_numeric_cells,
        non_finite_cells=non_finite_cells,
        invalid_category_cells=invalid_category_cells,
        invalid_binary_cells=invalid_binary_cells,
        non_positive_transform_cells=non_positive_transform_cells,
    )


def numeric_feature_frame(
    frame: pd.DataFrame,
    required_features: Sequence[str] = REQUIRED_FEATURES,
    category_mappings: Mapping[str, Mapping[str, Any]] | None = None,
    binary_features: Sequence[str] = BINARY_FEATURES,
) -> pd.DataFrame:
    """Return encoded model columns in model order after successful validation."""

    required = list(required_features)
    known_binary_fields = {str(feature) for feature in binary_features}
    numeric = pd.DataFrame(index=frame.index)
    for feature in required:
        mapping = _category_mapping_for(feature, category_mappings)
        if mapping is not None:
            encoded = encode_categorical_series(frame[feature], mapping)
            if encoded.isna().any():
                raise ValueError(f"{feature} contains invalid categories")
            numeric[feature] = encoded
        else:
            converted = pd.to_numeric(frame[feature], errors="raise").astype(float)
            if feature in known_binary_fields and not converted.isin((0.0, 1.0)).all():
                raise ValueError(f"{feature} must contain only 0 or 1")
            numeric[feature] = converted
    return numeric


def count_out_of_range(
    numeric_frame: pd.DataFrame,
    training_ranges: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, int]:
    """Count valid input values outside backend-supplied training ranges."""

    if not training_ranges:
        return {}

    counts: dict[str, int] = {}
    for feature in numeric_frame.columns:
        bounds = training_ranges.get(str(feature))
        if not isinstance(bounds, Mapping):
            continue
        try:
            lower = float(bounds["min"])
            upper = float(bounds["max"])
        except (KeyError, TypeError, ValueError):
            continue
        outside = (numeric_frame[feature] < lower) | (numeric_frame[feature] > upper)
        count = int(outside.sum())
        if count:
            counts[str(feature)] = count
    return counts


def inspect_model_bytes(contents: bytes) -> ModelInspection:
    """Load a trusted joblib package and expose its model feature names.

    Joblib is pickle-based, therefore callers must only pass files selected by
    a trusted user.  This function performs schema validation immediately so
    the actual training step cannot fail later due to an incomplete package.
    """

    bundle = joblib.load(BytesIO(contents))
    if not isinstance(bundle, dict):
        raise ValueError("The model file must be a joblib dictionary containing the model and preprocessors")
    required_keys = ("model", "scaler", "imputer")
    missing_keys = [key for key in required_keys if key not in bundle]
    if missing_keys:
        raise ValueError("The model package is missing keys: " + ", ".join(missing_keys))

    if "selected_features" not in bundle and "features" not in bundle:
        raise ValueError("The model package is missing selected_features or features")

    raw_features = bundle.get("selected_features", bundle.get("features"))
    if not isinstance(raw_features, (list, tuple)) or not raw_features:
        raise ValueError("Model selected_features is empty or invalid")
    features = tuple(str(feature) for feature in raw_features)
    if len(set(features)) != len(features):
        raise ValueError("Model selected_features contains duplicate fields")

    raw_categories = bundle.get("category_mappings", {})
    categories: dict[str, dict[str, float]] = {}
    if raw_categories:
        if not isinstance(raw_categories, Mapping):
            raise ValueError("Model category_mappings has an invalid format")
        for feature, raw_mapping in raw_categories.items():
            feature_name = str(feature)
            if feature_name not in features or not isinstance(raw_mapping, Mapping):
                raise ValueError("Model category_mappings contains an invalid feature")
            try:
                categories[feature_name] = {
                    str(label): float(code) for label, code in raw_mapping.items()
                }
            except (TypeError, ValueError) as error:
                raise ValueError("Model category_mappings contains an invalid encoding") from error

    raw_metadata = bundle.get("metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    raw_transforms = metadata.get("input_transforms", bundle.get("input_transforms", {}))
    if not isinstance(raw_transforms, Mapping):
        raise ValueError("Model input_transforms has an invalid format")
    transforms: dict[str, str] = {}
    for feature, transform in raw_transforms.items():
        feature_name = str(feature)
        transform_name = str(transform)
        if feature_name not in features or transform_name != "log10_positive":
            raise ValueError("Model input_transforms contains an invalid transform")
        transforms[feature_name] = transform_name

    raw_conditional = metadata.get("conditional_missing_when_a_zero", ())
    if not isinstance(raw_conditional, (list, tuple)):
        raise ValueError("Model conditional_missing_when_a_zero has an invalid format")
    conditional = tuple(
        str(feature)
        for feature in raw_conditional
        if str(feature) in features and str(feature) != "A"
    )

    model_type = type(bundle["model"]).__name__
    return ModelInspection(
        features=features,
        model_type=model_type,
        category_mappings=categories,
        input_transforms=transforms,
        conditional_missing_when_a_zero=conditional,
    )


def validate_fine_tune_frame(
    frame: pd.DataFrame,
    required_features: Sequence[str],
    category_mappings: Mapping[str, Mapping[str, Any]] | None = None,
    input_transforms: Mapping[str, str] | None = None,
    conditional_missing_when_a_zero: Sequence[str] = (),
) -> FineTuneValidation:
    """Validate feature columns and the positive linear-resistivity target."""

    if frame.shape[1] < 2:
        empty = FrameValidation(missing=tuple(required_features))
        return FineTuneValidation(frame=empty, row_count=len(frame))

    target_name = str(frame.columns[-1])
    feature_frame = frame.iloc[:, :-1]
    feature_validation = validate_model_input_frame(
        feature_frame,
        required_features,
        category_mappings,
        input_transforms=input_transforms,
        conditional_missing_when_a_zero=conditional_missing_when_a_zero,
    )
    target = frame.iloc[:, -1]
    text_empty = target.map(lambda value: isinstance(value, str) and not value.strip())
    empty_mask = target.isna() | text_empty
    numeric_target = pd.to_numeric(target, errors="coerce")
    non_numeric_mask = (~empty_mask) & numeric_target.isna()
    non_positive_mask = numeric_target.notna() & (
        ~np.isfinite(numeric_target.astype(float)) | (numeric_target.astype(float) <= 0)
    )

    return FineTuneValidation(
        frame=feature_validation,
        target_name=target_name,
        target_empty_cells=int(empty_mask.sum()),
        target_non_numeric_cells=int(non_numeric_mask.sum()),
        target_non_positive_cells=int(non_positive_mask.sum()),
        row_count=len(frame),
    )
