"""Adapt the supplied GWO-75 LSCO model to the public API contract.

The source joblib bundle keeps spreadsheet-style column names and was fitted
after applying ``log10`` to three positive oxygen-pressure fields.  The web
application deliberately exposes concise names and accepts pressure in mbar,
so this adapter records both the column translation and the input transform.
It does not retrain the model or change any tree weights.
"""

from __future__ import annotations

import argparse
import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib


MODEL_LABEL = "GWO-75"

CANONICAL_FEATURES: tuple[str, ...] = (
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

SOURCE_FEATURE_COLUMNS: tuple[str, ...] = (
    "Psyn (mbar)",
    "Oxygen activation",
    "Mismatch (%)",
    "Ts (deg C)",
    "A",
    "TA (deg C)",
    "PA (mbar)",
    "tA (h)",
    "Pc (mbar)",
    "t (nm)",
    "Sr (fraction)",
    "Tmeas (K)",
    "Growth method",
)

SOURCE_PRESSURE_COLUMNS: tuple[str, ...] = (
    "Psyn (mbar)",
    "PA (mbar)",
    "Pc (mbar)",
)

INPUT_TRANSFORMS: dict[str, str] = {
    "Psyn": "log10_positive",
    "PA": "log10_positive",
    "Pc": "log10_positive",
}

_SOURCE_TO_CANONICAL = dict(zip(SOURCE_FEATURE_COLUMNS, CANONICAL_FEATURES))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_category_mappings(raw: object) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        raise ValueError("The source model is missing category_mappings")

    output: dict[str, dict[str, float]] = {}
    for source_feature, raw_mapping in raw.items():
        canonical = _SOURCE_TO_CANONICAL.get(str(source_feature))
        if canonical is None:
            raise ValueError(f"Unknown categorical source field: {source_feature}")
        if not isinstance(raw_mapping, Mapping) or not raw_mapping:
            raise ValueError(f"Invalid category mapping for {source_feature}")
        output[canonical] = {
            str(label): float(code) for label, code in raw_mapping.items()
        }
    return output


def adopt_gwo75_model(
    source_path: str | Path,
    output_path: str | Path,
    *,
    source_target_unit: str,
) -> dict[str, Any]:
    """Write a non-destructive GWO-75 API adapter.

    ``source_target_unit`` is intentionally explicit because the source
    artifact does not contain target-unit metadata.  This application supports
    the confirmed Feuil5 convention ``ohm_cm`` and converts it to the public
    ``μΩ·cm`` log contract with a +6 offset.
    """

    if source_target_unit != "ohm_cm":
        raise ValueError("source_target_unit must be explicitly set to 'ohm_cm'")

    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source model not found: {source}")
    if destination.exists():
        raise FileExistsError(f"Output model already exists; refusing to overwrite: {destination}")
    if source == destination:
        raise ValueError("The adapter output cannot overwrite the source model")

    source_bundle = joblib.load(source)
    if not isinstance(source_bundle, dict):
        raise ValueError("The source model must be a joblib dictionary")
    required = (
        "model",
        "imputer",
        "scaler",
        "features",
        "valid_features",
        "category_mappings",
        "log_pressure",
        "pressure_columns",
    )
    missing = [key for key in required if key not in source_bundle]
    if missing:
        raise ValueError("The source model is missing keys: " + ", ".join(missing))

    source_features = tuple(str(value) for value in source_bundle["features"])
    valid_features = tuple(str(value) for value in source_bundle["valid_features"])
    if source_features != SOURCE_FEATURE_COLUMNS or valid_features != SOURCE_FEATURE_COLUMNS:
        raise ValueError(
            "The source feature order does not match the GWO-75 Feuil5 schema: "
            + ", ".join(source_features)
        )
    if int(getattr(source_bundle["model"], "n_features_in_", -1)) != len(CANONICAL_FEATURES):
        raise ValueError("The source estimator does not declare 13 input features")

    if source_bundle["log_pressure"] is not True:
        raise ValueError("The GWO-75 source bundle must declare log_pressure=True")
    pressure_columns = tuple(str(value) for value in source_bundle["pressure_columns"])
    if pressure_columns != SOURCE_PRESSURE_COLUMNS:
        raise ValueError(
            "Unexpected pressure_columns order: " + ", ".join(pressure_columns)
        )

    category_mappings = _canonical_category_mappings(
        source_bundle["category_mappings"]
    )
    expected_categories = {
        "Oxygen activation": {"No": 0.0, "Ozone": 1.0},
        "Growth method": {"MBE": 0.0, "PLD": 1.0},
    }
    if category_mappings != expected_categories:
        raise ValueError("The source categorical encodings do not match GWO-75")

    experiment_id = str(source_bundle.get("experiment_id") or "")
    if "GWO" not in experiment_id or int(source_bundle.get("pop_size", -1)) != 75:
        raise ValueError("The source artifact is not the expected GWO population-75 model")

    metadata = {
        "schema_version": 4,
        "model_label": MODEL_LABEL,
        "adapter_kind": "gwo75_feuil5_no_lattice_log_pressure",
        "created_utc": datetime.now(UTC).isoformat(),
        "source_model": source.name,
        "source_model_sha256": _sha256(source),
        "source_sheet": "Feuil5",
        "source_target_unit": "ohm_cm",
        "source_target_log_unit": "log10(Ω·cm)",
        "api_target_unit": "μΩ·cm",
        "api_target_log_unit": "log10(μΩ·cm)",
        "output_log_offset": 6.0,
        "target_conversion": "y_log_api = y_log_model + 6; y_api = 10 ** y_log_api",
        "selected_features": list(CANONICAL_FEATURES),
        "model_feature_columns": list(SOURCE_FEATURE_COLUMNS),
        "input_transforms": dict(INPUT_TRANSFORMS),
        "input_transform_note": "Psyn, PA, and Pc are entered in positive mbar and transformed with log10 before imputation and scaling.",
        "excluded_features": ["a", "c"],
        "conditional_missing_when_a_zero": ["TA", "PA", "tA"],
        "conditional_missing_note": "For A=0, TA/PA/tA are restored to missing values before saved-model preprocessing.",
        "experiment_id": experiment_id,
        "population_size": 75,
        "epochs_completed": int(source_bundle.get("epochs_completed", 0)),
        "dataset_sha256": str(source_bundle.get("dataset_sha256") or ""),
        "best_params": dict(source_bundle.get("best_params", {})),
    }
    adapted_bundle = {
        "model": source_bundle["model"],
        "imputer": source_bundle["imputer"],
        "scaler": source_bundle["scaler"],
        "selected_features": list(CANONICAL_FEATURES),
        "features": list(CANONICAL_FEATURES),
        "model_feature_columns": list(SOURCE_FEATURE_COLUMNS),
        "category_mappings": category_mappings,
        "input_transforms": dict(INPUT_TRANSFORMS),
        "metadata": metadata,
        "output_log_offset": 6.0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(adapted_bundle, destination)
    return {
        "output_path": str(destination),
        "model_label": MODEL_LABEL,
        "selected_features": list(CANONICAL_FEATURES),
        "input_transforms": dict(INPUT_TRANSFORMS),
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt the supplied GWO-75 Feuil5 model without retraining."
    )
    parser.add_argument("source_path", help="source GWO-75 .JOB/.joblib package")
    parser.add_argument("output_path", help="new adapter .joblib; must not exist")
    parser.add_argument(
        "--source-target-unit",
        required=True,
        choices=("ohm_cm",),
        help="confirmed source target unit; required because the source file omits it",
    )
    args = parser.parse_args()
    try:
        result = adopt_gwo75_model(
            args.source_path,
            args.output_path,
            source_target_unit=args.source_target_unit,
        )
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
