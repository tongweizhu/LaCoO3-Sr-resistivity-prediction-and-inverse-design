"""Tests for the standalone ten-feature all-data retraining command."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKSPACE_ROOT / "zhu"
sys.path.insert(0, str(BACKEND_ROOT))

from retrain_without_lattice import (  # noqa: E402
    FEATURES,
    MIN_ROWS,
    TrainingConfig,
    _build_argument_parser,
    prepare_training_data,
    retrain_without_lattice,
)
import backend_app  # noqa: E402


def _training_frame(condition_count: int = 6, temperatures_per_condition: int = 6) -> pd.DataFrame:
    """Build grouped synthetic data with source rho stored in Ω·cm."""

    rows: list[dict[str, float]] = []
    for condition in range(condition_count):
        for temperature_index in range(temperatures_per_condition):
            temperature = 20.0 + 45.0 * temperature_index
            values = {
                "Psyn": 100.0 + 30.0 * condition,
                "Ts": 650.0 + 10.0 * condition,
                "A": float(condition % 2),
                "TA": 350.0 + 5.0 * condition,
                "PA": 0.1 + 0.02 * condition,
                "tA": 0.02 + 0.005 * condition,
                "Pc": 10.0 + 25.0 * condition,
                "t": 30.0 + 8.0 * condition,
                "Sr": 0.05 + 0.04 * condition,
                "Tmeas": temperature,
                # Source-only lattice columns must not enter the artifact.
                "a": 3.80 + 0.001 * condition,
                "c": 3.90 + 0.002 * condition,
            }
            rho_uohm_cm = 1_200.0 + 16.0 * temperature + 180.0 * condition
            values["ρ"] = rho_uohm_cm / 1_000_000.0
            rows.append(values)
    return pd.DataFrame(rows)


class RetrainWithoutLatticeTest(unittest.TestCase):
    def test_retraining_creates_backend_compatible_ten_feature_bundle(self) -> None:
        frame = _training_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training.csv"
            output = root / "lsco_without_lattice.joblib"
            frame.to_csv(source, index=False)

            result = retrain_without_lattice(
                source,
                output,
                source_target_unit="ohm_cm",
                config=TrainingConfig(n_estimators=8, n_jobs=1),
            )

            self.assertTrue(output.is_file())
            self.assertEqual(result["selected_features"], list(FEATURES))
            self.assertEqual(result["excluded_features"], ["a", "c"])
            bundle = joblib.load(output)
            self.assertTrue({"model", "scaler", "imputer", "selected_features"}.issubset(bundle))
            self.assertEqual(bundle["selected_features"], list(FEATURES))
            self.assertNotIn("a", bundle["selected_features"])
            self.assertNotIn("c", bundle["selected_features"])
            self.assertEqual(list(bundle["imputer"].feature_names_in_), list(FEATURES))
            self.assertEqual(list(bundle["scaler"].feature_names_in_), list(FEATURES))
            self.assertEqual(bundle["metadata"]["source_target_unit"], "ohm_cm")
            self.assertEqual(bundle["metadata"]["api_target_unit"], "μΩ·cm")
            self.assertEqual(bundle["metadata"]["excluded_features"], ["a", "c"])
            self.assertEqual(bundle["metrics"]["cv_folds"], 5)

            values = frame.loc[:, list(FEATURES)]
            imputed = bundle["imputer"].transform(values)
            scaled = bundle["scaler"].transform(pd.DataFrame(imputed, columns=FEATURES))
            prediction_log = bundle["model"].predict(scaled)
            self.assertTrue(np.isfinite(prediction_log).all())

            # The newly written package can be loaded by the unmodified API
            # contract and no longer surfaces the old a/c schema notice.
            with (
                patch.object(backend_app, "saved", bundle),
                patch.object(backend_app, "model", bundle["model"]),
                patch.object(backend_app, "scaler", bundle["scaler"]),
                patch.object(backend_app, "imputer", bundle["imputer"]),
                patch.object(backend_app, "features", list(FEATURES)),
                patch.object(backend_app, "model_feature_columns", list(FEATURES)),
                patch.object(backend_app, "category_mappings", {}),
                patch.object(backend_app, "input_transforms", {}),
                patch.object(backend_app, "output_log_offset", 0.0),
                patch.object(backend_app, "conditional_missing_when_a_zero", []),
            ):
                response = backend_app.predict(
                    backend_app.PredictReq(
                        samples=[
                            {
                                feature: float(frame.loc[0, feature])
                                for feature in FEATURES
                            }
                        ]
                    )
                )
                metadata = backend_app.get_features()

            self.assertEqual(response.used_features, list(FEATURES))
            self.assertAlmostEqual(response.y[0], 10.0 ** response.y_log[0], places=8)
            self.assertEqual(metadata["features"], list(FEATURES))
            self.assertIsNone(metadata["schema_notice"])

    def test_source_target_unit_is_explicit_and_converted_to_uohm_cm(self) -> None:
        frame = _training_frame()
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "training.csv"
            frame.to_csv(source, index=False)
            prepared = prepare_training_data(source, source_target_unit="ohm_cm")
            with self.assertRaisesRegex(ValueError, "source_target_unit"):
                prepare_training_data(source, source_target_unit="not-a-unit")  # type: ignore[arg-type]

        expected = float(frame.loc[0, "ρ"]) * 1_000_000.0
        self.assertAlmostEqual(float(prepared.target_uohm_cm.iloc[0]), expected)

    def test_cli_requires_an_explicit_source_target_unit(self) -> None:
        parser = _build_argument_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["training.csv", "new_model.joblib"])

    def test_short_or_invalid_datasets_are_rejected_before_training(self) -> None:
        short_frame = _training_frame().iloc[: MIN_ROWS - 1].copy()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            short_source = root / "short.csv"
            short_frame.to_csv(short_source, index=False)
            with self.assertRaisesRegex(ValueError, str(MIN_ROWS)):
                prepare_training_data(short_source, source_target_unit="ohm_cm")

            invalid = _training_frame()
            invalid.loc[0, "A"] = 2.0
            invalid_source = root / "invalid.csv"
            invalid.to_csv(invalid_source, index=False)
            with self.assertRaisesRegex(ValueError, "A 必须"):
                prepare_training_data(invalid_source, source_target_unit="ohm_cm")

    def test_existing_output_is_never_overwritten(self) -> None:
        frame = _training_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "training.csv"
            output = root / "already_exists.joblib"
            frame.to_csv(source, index=False)
            output.write_bytes(b"keep-me")

            with self.assertRaisesRegex(FileExistsError, "拒绝覆盖"):
                retrain_without_lattice(
                    source,
                    output,
                    source_target_unit="ohm_cm",
                    config=TrainingConfig(n_estimators=8, n_jobs=1),
                )
            self.assertEqual(output.read_bytes(), b"keep-me")


if __name__ == "__main__":
    unittest.main()
