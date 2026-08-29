"""Regression coverage for the scientific UI's client-side validation rules."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = WORKSPACE_ROOT / "zhu" / "zhu"
sys.path.insert(0, str(APP_ROOT))

from components.validation import validate_required_numeric, validate_temperature_range  # noqa: E402
from workflow_utils import (  # noqa: E402
    REQUIRED_FEATURES,
    count_out_of_range,
    numeric_feature_frame,
    validate_fine_tune_frame,
    validate_model_input_frame,
)
from fine_tuning import _encode_model_features  # noqa: E402


CATEGORY_MAPPINGS = {
    "Oxygen activation": {"No": 0.0, "Ozone": 1.0},
    "Growth method": {"MBE": 0.0, "PLD": 1.0},
}
INPUT_TRANSFORMS = {
    "Psyn": "log10_positive",
    "PA": "log10_positive",
    "Pc": "log10_positive",
}
CONDITIONAL_FIELDS = ("TA", "PA", "tA")


def _valid_frame() -> pd.DataFrame:
    """One complete model sample, intentionally not in display order."""
    values = {
        "Tmeas": [100.0],
        "Sr": [0.0],
        "Growth method": ["PLD"],
        "t": [50.0],
        "Pc": [75.0],
        "tA": [0.01],
        "PA": [0.1],
        "TA": [300.0],
        "A": [1.0],
        "Ts": [800.0],
        "Mismatch": [1.0],
        "Oxygen activation": ["No"],
        "Psyn": [50000.0],
    }
    return pd.DataFrame(values)


class SinglePredictionValidationTest(unittest.TestCase):
    def test_complete_numeric_values_and_binary_a_are_accepted(self) -> None:
        values = {feature: "1" for feature in REQUIRED_FEATURES}
        values["A"] = "0"
        values["Oxygen activation"] = "No"
        values["Growth method"] = "PLD"
        self.assertEqual(
            validate_required_numeric(
                values,
                REQUIRED_FEATURES,
                category_mappings=CATEGORY_MAPPINGS,
                input_transforms=INPUT_TRANSFORMS,
                conditional_missing_when_a_zero=CONDITIONAL_FIELDS,
            ),
            {},
        )

    def test_log_pressure_requires_positive_values_except_unannealed_placeholders(self) -> None:
        values = {feature: "1" for feature in REQUIRED_FEATURES}
        values.update({"Oxygen activation": "No", "Growth method": "MBE", "Psyn": "0"})
        errors = validate_required_numeric(
            values,
            REQUIRED_FEATURES,
            category_mappings=CATEGORY_MAPPINGS,
            input_transforms=INPUT_TRANSFORMS,
            conditional_missing_when_a_zero=CONDITIONAL_FIELDS,
        )
        self.assertIn("Psyn", errors)

        values.update({"Psyn": "1e-5", "A": "0", "PA": "0"})
        self.assertEqual(
            validate_required_numeric(
                values,
                REQUIRED_FEATURES,
                category_mappings=CATEGORY_MAPPINGS,
                input_transforms=INPUT_TRANSFORMS,
                conditional_missing_when_a_zero=CONDITIONAL_FIELDS,
            ),
            {},
        )

    def test_missing_nonfinite_and_invalid_binary_are_rejected(self) -> None:
        values = {feature: "1" for feature in REQUIRED_FEATURES}
        values.update({"Psyn": "", "Ts": "nan", "A": "2", "Oxygen activation": "air"})
        errors = validate_required_numeric(values, REQUIRED_FEATURES, category_mappings=CATEGORY_MAPPINGS)
        self.assertIn("Psyn", errors)
        self.assertIn("Ts", errors)
        self.assertIn("A", errors)
        self.assertIn("Oxygen activation", errors)

    def test_temperature_requires_an_increasing_finite_interval(self) -> None:
        self.assertEqual(validate_temperature_range("3", "345"), (True, ""))
        self.assertFalse(validate_temperature_range("345", "3")[0])
        self.assertFalse(validate_temperature_range("nan", "345")[0])


class BatchAndFineTuneValidationTest(unittest.TestCase):
    def test_batch_columns_can_arrive_in_any_order_and_extra_columns_are_retained(self) -> None:
        frame = _valid_frame()
        frame["operator_note"] = ["sample A"]
        validation = validate_model_input_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS)

        self.assertTrue(validation.is_valid)
        self.assertEqual(validation.missing, ())
        self.assertEqual(validation.extras, ("operator_note",))
        self.assertEqual(
            list(numeric_feature_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS).columns),
            list(REQUIRED_FEATURES),
        )

    def test_batch_blocks_missing_empty_and_non_numeric_required_cells(self) -> None:
        frame = _valid_frame().drop(columns=["Pc"]).astype({"TA": "object", "Ts": "object"})
        frame.loc[0, "TA"] = ""
        frame.loc[0, "Ts"] = "not-a-number"
        validation = validate_model_input_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS)

        self.assertFalse(validation.is_valid)
        self.assertIn("Pc", validation.missing)
        self.assertEqual(validation.empty_cells["TA"], 1)
        self.assertEqual(validation.non_numeric_cells["Ts"], 1)

    def test_batch_and_fine_tune_reject_non_binary_a(self) -> None:
        frame = _valid_frame()
        frame.loc[0, "A"] = 2.0
        validation = validate_model_input_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS)

        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.invalid_binary_cells["A"], 1)
        self.assertFalse(
            validate_fine_tune_frame(
                frame.assign(rho_uohm_cm=1000.0),
                REQUIRED_FEATURES,
                CATEGORY_MAPPINGS,
            ).is_valid
        )

    def test_fine_tune_encoding_matches_unannealed_prediction_semantics(self) -> None:
        frame = _valid_frame()
        frame.loc[0, "A"] = 0.0
        frame.loc[0, ["TA", "PA", "tA"]] = 0.0
        encoded = _encode_model_features(
            frame,
            list(REQUIRED_FEATURES),
            CATEGORY_MAPPINGS,
            CONDITIONAL_FIELDS,
            INPUT_TRANSFORMS,
        )

        self.assertEqual(encoded.loc[0, "Oxygen activation"], 0.0)
        self.assertEqual(encoded.loc[0, "Growth method"], 1.0)
        self.assertTrue(np.isnan(encoded.loc[0, "TA"]))
        self.assertTrue(np.isnan(encoded.loc[0, "PA"]))
        self.assertTrue(np.isnan(encoded.loc[0, "tA"]))
        self.assertAlmostEqual(encoded.loc[0, "Psyn"], np.log10(50000.0))
        self.assertAlmostEqual(encoded.loc[0, "Pc"], np.log10(75.0))

    def test_batch_rejects_non_positive_transformed_pressure(self) -> None:
        frame = _valid_frame()
        frame.loc[0, "Psyn"] = 0.0
        validation = validate_model_input_frame(
            frame,
            REQUIRED_FEATURES,
            CATEGORY_MAPPINGS,
            input_transforms=INPUT_TRANSFORMS,
            conditional_missing_when_a_zero=CONDITIONAL_FIELDS,
        )
        self.assertFalse(validation.is_valid)
        self.assertEqual(validation.non_positive_transform_cells["Psyn"], 1)

    def test_out_of_range_count_is_reported_without_reordering(self) -> None:
        frame = numeric_feature_frame(_valid_frame(), REQUIRED_FEATURES, CATEGORY_MAPPINGS)
        frame.loc[0, "Tmeas"] = 500.0
        counts = count_out_of_range(
            frame,
            {"Tmeas": {"min": 2.65, "max": 344.71}},
        )
        self.assertEqual(counts, {"Tmeas": 1})

    def test_fine_tune_requires_positive_last_target_column(self) -> None:
        frame = _valid_frame()
        frame["rho_uohm_cm"] = [0.0]
        invalid = validate_fine_tune_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS)
        self.assertFalse(invalid.is_valid)
        self.assertEqual(invalid.target_non_positive_cells, 1)

        frame["rho_uohm_cm"] = [1000.0]
        valid = validate_fine_tune_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS)
        self.assertFalse(valid.is_valid, "one row is insufficient for metric evaluation")
        frame = pd.concat([frame, frame], ignore_index=True)
        self.assertTrue(validate_fine_tune_frame(frame, REQUIRED_FEATURES, CATEGORY_MAPPINGS).is_valid)


if __name__ == "__main__":
    unittest.main()
