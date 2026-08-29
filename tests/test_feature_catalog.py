"""Regression coverage for the user-confirmed LSCO presentation vocabulary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "zhu"))

from zhu.components.feature_catalog import (  # noqa: E402
    ACTIVE_UI_FIELDS,
    FEATURE_GROUPS,
    REMOVED_FIELDS,
    RHO_OUTPUT,
    get_feature_hint,
    get_feature_label,
)


class FeatureCatalogTest(unittest.TestCase):
    def test_pop50_schema_excludes_lattice_parameters(self) -> None:
        active_keys = {field.key for field in ACTIVE_UI_FIELDS}
        self.assertEqual(
            active_keys,
            {
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
            },
        )
        self.assertEqual({field.key for field in REMOVED_FIELDS}, {"a", "c"})
        self.assertNotIn("a", active_keys)
        self.assertNotIn("c", active_keys)

    def test_annealing_and_measurement_labels_keep_confirmed_units(self) -> None:
        self.assertEqual(get_feature_label("A"), "A · Annealing status")
        self.assertIn("Options: 0 = not annealed", get_feature_hint("A"))
        self.assertNotIn("未退火", get_feature_hint("A"))
        self.assertEqual(get_feature_label("Tmeas"), "Tmeas · Measurement temperature (K)")
        self.assertEqual(FEATURE_GROUPS["退火条件"], ("A", "TA", "PA", "tA"))
        self.assertIn("Oxygen activation", FEATURE_GROUPS["制备条件"])

    def test_resistivity_catalog_distinguishes_source_and_api_units(self) -> None:
        self.assertEqual(RHO_OUTPUT.unit, "Ω·cm")
        self.assertEqual(RHO_OUTPUT.api_unit, "μΩ·cm")


if __name__ == "__main__":
    unittest.main()
