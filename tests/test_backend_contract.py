"""Regression coverage for the API's resistivity-unit contract."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from fastapi import HTTPException


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKSPACE_ROOT / "zhu"
sys.path.insert(0, str(BACKEND_ROOT))

import backend_app  # noqa: E402


class BackendUnitContractTest(unittest.TestCase):
    """Ensure API output stays in the model training label's μΩ·cm unit."""

    sample = {
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
    }

    def test_linear_prediction_is_ten_to_the_logged_prediction(self) -> None:
        result = backend_app.predict(backend_app.PredictReq(samples=[self.sample]))

        self.assertEqual(result.y_unit, "μΩ·cm")
        self.assertEqual(result.output_transform, "y = 10 ** y_log")
        self.assertAlmostEqual(result.y[0], 10.0 ** result.y_log[0], places=8)
        # GWO-75 was trained on Ω·cm. The adapter shifts log10 by +6, so the
        # public response stays in μΩ·cm while retaining y = 10 ** y_log.
        self.assertAlmostEqual(result.y[0], 1242.1987113392463, places=6)
        self.assertEqual(result.used_features, backend_app.features)

    def test_prediction_matches_manual_log_pressure_preprocessing(self) -> None:
        frame = pd.DataFrame([self.sample]).reindex(columns=backend_app.features)
        numeric, _, _, _ = backend_app._encode_input_frame(frame)
        for feature in ("Psyn", "PA", "Pc"):
            numeric[feature] = np.log10(numeric[feature])
        numeric.columns = backend_app.model_feature_columns
        imputed = backend_app.imputer.transform(numeric)
        imputed_frame = pd.DataFrame(
            imputed,
            columns=backend_app.model_feature_columns,
            index=numeric.index,
        )
        scaled = backend_app.scaler.transform(
            imputed_frame if hasattr(backend_app.scaler, "feature_names_in_") else imputed
        )
        expected_log = float(backend_app.model.predict(scaled)[0]) + 6.0

        result = backend_app.predict(backend_app.PredictReq(samples=[self.sample]))
        self.assertAlmostEqual(result.y_log[0], expected_log, places=10)

    def test_non_positive_log_pressure_is_rejected(self) -> None:
        invalid = dict(self.sample, Psyn=0.0)
        with self.assertRaises(HTTPException) as raised:
            backend_app.predict(backend_app.PredictReq(samples=[invalid]))
        self.assertEqual(getattr(raised.exception, "status_code", None), 400)
        self.assertIn("positive", str(getattr(raised.exception, "detail", "")))

    def test_health_endpoint_declares_the_same_unit(self) -> None:
        health = backend_app.healthz()
        self.assertEqual(health["status"], "ok")
        self.assertEqual(health["units"]["y"], "μΩ·cm")
        self.assertEqual(health["units"]["output_transform"], "y = 10 ** y_log")
        self.assertEqual(
            health["input_transforms"],
            {"Psyn": "log10_positive", "PA": "log10_positive", "Pc": "log10_positive"},
        )

    def test_out_of_training_range_inputs_are_reported(self) -> None:
        extrapolation_sample = dict(self.sample, Ts=2_000.0)
        result = backend_app.predict(backend_app.PredictReq(samples=[extrapolation_sample]))

        input_diagnostics = result.diagnostics["input"]
        self.assertIn("Ts", input_diagnostics["out_of_training_range_features"])
        self.assertEqual(input_diagnostics["out_of_training_range_counts"]["Ts"], 1)

    def test_category_labels_and_numeric_codes_are_equivalent(self) -> None:
        labels = backend_app.predict(backend_app.PredictReq(samples=[self.sample]))
        encoded_sample = dict(self.sample, **{"Oxygen activation": 0, "Growth method": 0})
        codes = backend_app.predict(backend_app.PredictReq(samples=[encoded_sample]))

        self.assertAlmostEqual(labels.y_log[0], codes.y_log[0], places=8)
        self.assertEqual(labels.diagnostics["input"]["invalid_category_counts"], {
            feature: 0 for feature in backend_app.features
        })

    def test_unannealed_zero_placeholders_match_trained_missing_value_semantics(self) -> None:
        placeholder_sample = dict(self.sample, A=0, TA=0, PA=0, tA=0)
        placeholders = backend_app.predict(backend_app.PredictReq(samples=[placeholder_sample]))
        source_style_missing = dict(placeholder_sample, TA=None, PA=None, tA=None)
        missing = backend_app.predict(backend_app.PredictReq(samples=[source_style_missing]))

        self.assertAlmostEqual(placeholders.y_log[0], missing.y_log[0], places=8)
        self.assertEqual(
            placeholders.diagnostics["input"]["conditional_missing_counts"],
            {"TA": 1, "PA": 1, "tA": 1},
        )


class BackendFeatureMetadataTest(unittest.TestCase):
    """Ensure /features stays useful for both loaded and failed models."""

    def test_features_metadata_describes_the_loaded_model(self) -> None:
        metadata = backend_app.get_features()

        self.assertEqual(metadata["features"], backend_app.features)
        self.assertEqual(metadata["temperature_feature"], "Tmeas")
        self.assertEqual(metadata["input_unit_note"], backend_app.INPUT_UNIT_NOTE)
        self.assertEqual(metadata["target_unit"], "μΩ·cm")
        self.assertEqual(metadata["target_log_unit"], "log10(μΩ·cm)")
        self.assertEqual(metadata["output_transform"], "y = 10 ** y_log")
        self.assertEqual(metadata["model_label"], "GWO-75")
        self.assertNotIn(str(Path(backend_app.MODEL_PATH).parent), metadata["model_label"])
        self.assertIsNone(metadata["warning"])
        self.assertIsNone(metadata["schema_notice"])
        self.assertNotIn("a", metadata["features"])
        self.assertNotIn("c", metadata["features"])
        self.assertEqual(metadata["source_target_unit"], "ohm_cm")
        self.assertEqual(metadata["output_log_offset"], 6.0)
        self.assertEqual(
            metadata["input_transforms"],
            {"Psyn": "log10_positive", "PA": "log10_positive", "Pc": "log10_positive"},
        )
        self.assertEqual(
            metadata["categorical_features"],
            {
                "Oxygen activation": {"No": 0.0, "Ozone": 1.0},
                "Growth method": {"MBE": 0.0, "PLD": 1.0},
            },
        )

        ranges = metadata["training_ranges"]
        self.assertEqual(set(ranges), set(backend_app.features))
        for feature, lower, upper in zip(
            backend_app.features,
            backend_app.scaler.data_min_,
            backend_app.scaler.data_max_,
        ):
            if metadata["input_transforms"].get(feature) == "log10_positive":
                lower = 10.0 ** float(lower)
                upper = 10.0 ** float(upper)
            self.assertEqual(ranges[feature], {"min": float(lower), "max": float(upper)})

    def test_temperature_feature_is_null_when_exact_tmeas_is_absent(self) -> None:
        with patch.object(backend_app, "features", ["Psyn", "T_meas"]):
            metadata = backend_app.get_features()

        self.assertIsNone(metadata["temperature_feature"])

    def test_legacy_lattice_schema_is_flagged_when_present(self) -> None:
        with patch.object(backend_app, "features", ["Psyn", "a", "c", "Tmeas"]):
            metadata = backend_app.get_features()

            self.assertIn("a, c", metadata["schema_notice"] or "")

    def test_unloaded_model_returns_stable_metadata_with_warning(self) -> None:
        with (
            patch.object(backend_app, "saved", None),
            patch.object(backend_app, "model", None),
            patch.object(backend_app, "scaler", None),
            patch.object(backend_app, "imputer", None),
            patch.object(backend_app, "features", []),
            patch.object(backend_app, "model_load_warning", "模型未正确加载；特征元数据与预测暂不可用。"),
        ):
            metadata = backend_app.get_features()

        self.assertEqual(metadata["features"], [])
        self.assertIsNone(metadata["temperature_feature"])
        self.assertEqual(metadata["training_ranges"], {})
        self.assertEqual(metadata["model_label"], Path(backend_app.MODEL_PATH).name)
        self.assertIsNone(metadata["schema_notice"])
        self.assertTrue(metadata["warning"])
        self.assertEqual(metadata["target_unit"], "μΩ·cm")
        self.assertEqual(metadata["output_transform"], "y = 10 ** y_log")


if __name__ == "__main__":
    unittest.main()
