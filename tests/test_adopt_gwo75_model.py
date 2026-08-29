"""Tests for the non-destructive GWO-75 API model adapter."""

from __future__ import annotations

import hashlib
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyRegressor
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKSPACE_ROOT / "zhu"
sys.path.insert(0, str(BACKEND_ROOT))

from adopt_gwo75_model import (  # noqa: E402
    CANONICAL_FEATURES,
    INPUT_TRANSFORMS,
    SOURCE_FEATURE_COLUMNS,
    SOURCE_PRESSURE_COLUMNS,
    adopt_gwo75_model,
)


REQUIRED_SOURCE_KEYS = (
    "model",
    "imputer",
    "scaler",
    "features",
    "valid_features",
    "category_mappings",
    "log_pressure",
    "pressure_columns",
)


def _valid_source_bundle() -> dict[str, Any]:
    """Return a small, fitted, fully serializable GWO-75-shaped bundle."""

    rows = np.vstack(
        (
            np.arange(1.0, len(SOURCE_FEATURE_COLUMNS) + 1.0),
            np.arange(2.0, len(SOURCE_FEATURE_COLUMNS) + 2.0),
            np.arange(3.0, len(SOURCE_FEATURE_COLUMNS) + 3.0),
        )
    )
    frame = pd.DataFrame(rows, columns=SOURCE_FEATURE_COLUMNS)
    imputer = SimpleImputer(strategy="mean").fit(frame)
    imputed = imputer.transform(frame)
    scaler = MinMaxScaler().fit(imputed)
    model = DummyRegressor(strategy="mean").fit(
        scaler.transform(imputed),
        np.array([-3.0, -2.8, -2.6]),
    )
    return {
        "model": model,
        "imputer": imputer,
        "scaler": scaler,
        "features": list(SOURCE_FEATURE_COLUMNS),
        "valid_features": list(SOURCE_FEATURE_COLUMNS),
        "category_mappings": {
            "Oxygen activation": {"No": 0, "Ozone": 1},
            "Growth method": {"MBE": 0, "PLD": 1},
        },
        "log_pressure": True,
        "pressure_columns": list(SOURCE_PRESSURE_COLUMNS),
        "experiment_id": "GWO-XGBoost_Feuil5_pop75_epoch200_test",
        "pop_size": 75,
        "epochs_completed": 200,
        "dataset_sha256": "synthetic-dataset-sha256",
        "best_params": {"n_estimators": 989, "max_depth": 6},
    }


def _dump_source(path: Path, bundle: dict[str, Any] | None = None) -> dict[str, Any]:
    source_bundle = _valid_source_bundle() if bundle is None else bundle
    joblib.dump(source_bundle, path)
    return source_bundle


class AdoptGwo75ModelTest(unittest.TestCase):
    def test_adapter_records_exact_schema_transforms_units_and_source_hash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "MODEL_GWO75.JOB"
            output = root / "LSCO_gwo75_api_v1.joblib"
            _dump_source(source)
            expected_source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()

            result = adopt_gwo75_model(
                source,
                output,
                source_target_unit="ohm_cm",
            )

            self.assertTrue(output.is_file())
            self.assertEqual(result["selected_features"], list(CANONICAL_FEATURES))
            self.assertEqual(result["input_transforms"], INPUT_TRANSFORMS)

            bundle = joblib.load(output)
            metadata = bundle["metadata"]
            self.assertEqual(bundle["selected_features"], list(CANONICAL_FEATURES))
            self.assertEqual(bundle["features"], list(CANONICAL_FEATURES))
            self.assertEqual(
                bundle["model_feature_columns"],
                list(SOURCE_FEATURE_COLUMNS),
            )
            self.assertEqual(
                dict(zip(bundle["model_feature_columns"], bundle["selected_features"])),
                dict(zip(SOURCE_FEATURE_COLUMNS, CANONICAL_FEATURES)),
            )
            self.assertEqual(
                bundle["category_mappings"],
                {
                    "Oxygen activation": {"No": 0.0, "Ozone": 1.0},
                    "Growth method": {"MBE": 0.0, "PLD": 1.0},
                },
            )
            self.assertEqual(bundle["input_transforms"], INPUT_TRANSFORMS)
            self.assertEqual(metadata["input_transforms"], INPUT_TRANSFORMS)
            self.assertEqual(metadata["source_target_unit"], "ohm_cm")
            self.assertEqual(metadata["source_target_log_unit"], "log10(Ω·cm)")
            self.assertEqual(metadata["api_target_unit"], "μΩ·cm")
            self.assertEqual(metadata["api_target_log_unit"], "log10(μΩ·cm)")
            self.assertEqual(bundle["output_log_offset"], 6.0)
            self.assertEqual(metadata["output_log_offset"], 6.0)
            self.assertEqual(metadata["source_model_sha256"], expected_source_sha256)
            self.assertEqual(metadata["excluded_features"], ["a", "c"])
            self.assertNotIn("a", bundle["selected_features"])
            self.assertNotIn("c", bundle["selected_features"])
            self.assertEqual(metadata["population_size"], 75)
            self.assertEqual(metadata["experiment_id"], "GWO-XGBoost_Feuil5_pop75_epoch200_test")

    def test_wrong_feature_order_is_rejected(self) -> None:
        for field in ("features", "valid_features"):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "wrong_order.JOB"
                output = root / "adapter.joblib"
                bundle = _valid_source_bundle()
                bundle[field][0], bundle[field][1] = bundle[field][1], bundle[field][0]
                _dump_source(source, bundle)

                with self.assertRaisesRegex(ValueError, "feature order"):
                    adopt_gwo75_model(
                        source,
                        output,
                        source_target_unit="ohm_cm",
                    )
                self.assertFalse(output.exists())

    def test_wrong_pressure_columns_are_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "wrong_pressure_columns.JOB"
            output = root / "adapter.joblib"
            bundle = _valid_source_bundle()
            bundle["pressure_columns"] = list(reversed(SOURCE_PRESSURE_COLUMNS))
            _dump_source(source, bundle)

            with self.assertRaisesRegex(ValueError, "pressure_columns"):
                adopt_gwo75_model(
                    source,
                    output,
                    source_target_unit="ohm_cm",
                )
            self.assertFalse(output.exists())

    def test_each_required_source_key_is_enforced(self) -> None:
        for missing_key in REQUIRED_SOURCE_KEYS:
            with self.subTest(missing_key=missing_key), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / f"missing_{missing_key}.JOB"
                output = root / "adapter.joblib"
                bundle = _valid_source_bundle()
                del bundle[missing_key]
                _dump_source(source, bundle)

                with self.assertRaisesRegex(ValueError, "missing keys"):
                    adopt_gwo75_model(
                        source,
                        output,
                        source_target_unit="ohm_cm",
                    )
                self.assertFalse(output.exists())

    def test_existing_output_is_never_overwritten(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "valid_source.JOB"
            output = root / "existing.joblib"
            _dump_source(source)
            output.write_bytes(b"keep-me")

            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                adopt_gwo75_model(
                    source,
                    output,
                    source_target_unit="ohm_cm",
                )
            self.assertEqual(output.read_bytes(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
