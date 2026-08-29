"""Adapt the supplied Pop50 LSCO model to this application's API contract.

The source package was trained from Feuil5 without ``a`` and ``c``.  Its
internal feature names are spreadsheet headers and its target is
``log10(ρ / (Ω·cm))``.  The web/API contract uses concise feature names and
returns ``μΩ·cm``.  This tool writes a new, non-destructive adapter bundle
which records both translations; it does not retrain or alter tree weights.
"""

from __future__ import annotations

import argparse
import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Mapping

import joblib


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
    "Ts (°C)",
    "A",
    "TA (°C)",
    "PA (mbar)",
    "tA (h)",
    "Pc (mbar)",
    "t (nm)",
    "Sr",
    "Tmeas (K)",
    "Growth method",
)

_SOURCE_TO_CANONICAL = dict(zip(SOURCE_FEATURE_COLUMNS, CANONICAL_FEATURES))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_category_mappings(
    raw: object,
) -> dict[str, dict[str, float]]:
    if not isinstance(raw, Mapping):
        raise ValueError("源模型缺少有效 category_mappings")

    output: dict[str, dict[str, float]] = {}
    for source_feature, mapping in raw.items():
        canonical = _SOURCE_TO_CANONICAL.get(str(source_feature))
        if canonical is None:
            raise ValueError(f"源模型类别字段不在已知 Feuil5 模式中: {source_feature}")
        if not isinstance(mapping, Mapping) or not mapping:
            raise ValueError(f"源模型类别字段 {source_feature} 的编码无效")
        output[canonical] = {str(label): float(code) for label, code in mapping.items()}
    return output


def adopt_pop50_model(source_path: str | Path, output_path: str | Path) -> dict[str, Any]:
    """Write an API-compatible adapter without modifying the source package."""

    source = Path(source_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"源模型不存在: {source}")
    if destination.exists():
        raise FileExistsError(f"输出模型已存在，拒绝覆盖: {destination}")
    if source == destination:
        raise ValueError("输出模型不能覆盖源模型")

    source_bundle = joblib.load(source)
    if not isinstance(source_bundle, dict):
        raise ValueError("源模型必须是 joblib 字典")
    required = ("model", "imputer", "scaler", "features", "category_mappings")
    missing = [key for key in required if key not in source_bundle]
    if missing:
        raise ValueError("源模型缺少键: " + "、".join(missing))

    source_features = tuple(str(value) for value in source_bundle["features"])
    if source_features != SOURCE_FEATURE_COLUMNS:
        raise ValueError(
            "源模型特征顺序与 Pop50 Feuil5 模式不一致；实际为: "
            + "、".join(source_features)
        )

    category_mappings = _canonical_category_mappings(source_bundle["category_mappings"])
    if set(category_mappings) != {"Oxygen activation", "Growth method"}:
        raise ValueError("源模型的类别字段应为 Oxygen activation 与 Growth method")

    metadata = {
        "schema_version": 3,
        "model_label": destination.name,
        "adapter_kind": "pop50_feuil5_no_lattice",
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
        "conditional_missing_when_a_zero": ["TA", "PA", "tA"],
        "conditional_missing_note": (
            "For A=0, Feuil5 stores TA/PA/tA as not applicable blanks; "
            "the adapter preserves the saved imputer behavior."
        ),
        "selected_features": list(CANONICAL_FEATURES),
        "model_feature_columns": list(SOURCE_FEATURE_COLUMNS),
        "excluded_features": ["a", "c"],
    }
    adapted_bundle = {
        "model": source_bundle["model"],
        "imputer": source_bundle["imputer"],
        "scaler": source_bundle["scaler"],
        "selected_features": list(CANONICAL_FEATURES),
        "features": list(CANONICAL_FEATURES),
        "model_feature_columns": list(SOURCE_FEATURE_COLUMNS),
        "category_mappings": category_mappings,
        "metadata": metadata,
        "output_log_offset": 6.0,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(adapted_bundle, destination)
    return {
        "output_path": str(destination),
        "selected_features": list(CANONICAL_FEATURES),
        "category_mappings": category_mappings,
        "metadata": metadata,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Adapt supplied Pop50 Feuil5 model for the LSCO web API without retraining."
    )
    parser.add_argument("source_path", help="source pop50/model_pipeline.joblib")
    parser.add_argument("output_path", help="new adapter .joblib; must not already exist")
    args = parser.parse_args()
    try:
        result = adopt_pop50_model(args.source_path, args.output_path)
    except (FileNotFoundError, FileExistsError, ValueError) as error:
        parser.error(str(error))
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
