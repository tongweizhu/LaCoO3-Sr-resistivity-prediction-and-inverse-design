"""Coverage for the bounded LSCO inverse-design API and pure search helpers."""

from __future__ import annotations

from dataclasses import replace
import math
import sys
import unittest
from pathlib import Path

import pandas as pd
from fastapi import HTTPException


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
BACKEND_ROOT = WORKSPACE_ROOT / "zhu"
sys.path.insert(0, str(BACKEND_ROOT))

import backend_app  # noqa: E402
from inverse_design import (  # noqa: E402
    InverseDesignConfig,
    MAX_INVERSE_CANDIDATES,
    build_candidate_grid,
    dataframe_records,
    mismatch_from_sr_substrate,
    mismatch_table_records,
    normalized_config,
    rank_candidates,
    validate_inverse_config,
)


class InverseDesignPureFunctionTest(unittest.TestCase):
    def test_default_grid_matches_the_developer_package(self) -> None:
        config = InverseDesignConfig()
        grid = build_candidate_grid(config)

        self.assertEqual(len(grid), 2_000)
        self.assertEqual(validate_inverse_config(config), 2_000)
        self.assertEqual(
            list(grid.columns),
            backend_app.features + ["Substrate"],
        )
        sto_half_doped = grid.loc[
            (grid["Sr"] == 0.5) & (grid["Substrate"] == "STO"), "Mismatch"
        ]
        self.assertTrue((sto_half_doped == 1.96).all())
        self.assertNotIn("Substrate", backend_app.features)

    def test_mismatch_interpolates_only_inside_the_declared_domain(self) -> None:
        self.assertAlmostEqual(mismatch_from_sr_substrate(0.5, "LAO"), -1.07)
        midpoint = mismatch_from_sr_substrate(0.55, "STO")
        self.assertAlmostEqual(midpoint, (1.96 + 1.91) / 2.0)
        with self.assertRaisesRegex(ValueError, "0..0.8"):
            mismatch_from_sr_substrate(0.81, "STO")
        with self.assertRaisesRegex(ValueError, "Unknown substrate"):
            mismatch_from_sr_substrate(0.5, "unknown")

    def test_strict_validation_rejects_invalid_inputs(self) -> None:
        base = InverseDesignConfig()
        invalid_configs = (
            replace(base, target_rho_uohm_cm=math.nan),
            replace(base, target_rho_uohm_cm=0.0),
            replace(base, sr_values=()),
            replace(base, sr_values=(0.81,)),
            replace(base, ts_values=(math.inf,)),
            replace(base, tmeas_values=(0.0,)),
            replace(base, substrates=()),
            replace(base, substrates=("Si",)),
            replace(base, psyn_mbar_values=(0.0,)),
            replace(base, pc_mbar_values=(-1.0,)),
            replace(base, pa_mbar=0.0),
            replace(base, ta_h=-1.0),
            replace(base, thickness_nm=0.0),
            replace(base, annealing_values=()),
            replace(base, annealing_values=(2,)),
            replace(base, growth_method="CVD"),
            replace(base, oxygen_activation="Yes"),
            replace(base, top_n=0),
            replace(base, top_n=101),
        )
        for config in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    validate_inverse_config(config)

    def test_candidate_limit_is_checked_before_grid_construction(self) -> None:
        config = replace(
            InverseDesignConfig(),
            sr_values=(0.5,) * (MAX_INVERSE_CANDIDATES + 1),
            ts_values=(700.0,),
            tmeas_values=(300.0,),
            substrates=("STO",),
            psyn_mbar_values=(1e-5,),
            pc_mbar_values=(1e-5,),
            annealing_values=(1,),
        )
        with self.assertRaisesRegex(ValueError, "maximum is 20000"):
            build_candidate_grid(config)

    def test_ranking_uses_log_error_then_relative_error(self) -> None:
        candidates = pd.DataFrame({"candidate": ["high", "exact", "low"]})
        predictions = [1000.0, 100.0, 10.0]
        logs = [math.log10(value) for value in predictions]
        ranked = rank_candidates(candidates, predictions, logs, 100.0)

        self.assertEqual(ranked["candidate"].tolist(), ["exact", "low", "high"])
        self.assertEqual(ranked["Rank"].tolist(), [1, 2, 3])
        self.assertEqual(ranked.iloc[0]["Abs_error_log10"], 0.0)

    def test_download_support_records_are_json_safe_and_path_free(self) -> None:
        records = dataframe_records(pd.DataFrame({"value": [1.0, float("nan")]}))
        self.assertEqual(records, [{"value": 1.0}, {"value": None}])

        config = normalized_config(InverseDesignConfig())
        self.assertEqual(config["target_unit"], "μΩ·cm")
        self.assertEqual(config["candidate_count"], 2_000)
        self.assertFalse(any("path" in key.lower() for key in config))
        self.assertEqual(len(mismatch_table_records()), 13)


class InverseDesignEndpointTest(unittest.TestCase):
    def test_default_search_reproduces_the_packaged_smoke_result(self) -> None:
        result = backend_app.run_inverse_design(backend_app.InverseDesignReq())

        self.assertEqual(result.candidate_count, 2_000)
        self.assertEqual(len(result.all_candidates), 2_000)
        self.assertEqual(len(result.top_candidates), 20)
        self.assertEqual(result.target_unit, "μΩ·cm")
        self.assertEqual(result.model_label, "GWO-75")
        self.assertIn("not a unique", result.interpretation)
        self.assertEqual(len(result.mismatch_table), 13)
        self.assertEqual(result.normalized_config["candidate_limit"], 20_000)

        best = result.top_candidates[0]
        self.assertAlmostEqual(best["Predicted_rho_uohm_cm"], 853.145532, places=3)
        self.assertEqual(best["Substrate"], "LAO")
        self.assertEqual(best["Sr"], 0.5)
        self.assertEqual(best["Ts"], 700.0)
        self.assertAlmostEqual(best["Psyn"], 0.2 * 1.333223684)
        self.assertAlmostEqual(best["Pc"], 0.01 * 1.333223684)

    def test_registered_route_serializes_unannealed_missing_fields_as_null(self) -> None:
        self.assertIn("/inverse-design", {route.path for route in backend_app.app.routes})
        result = backend_app.run_inverse_design(
            backend_app.InverseDesignReq(
                target_rho_uohm_cm=1_200.0,
                top_n=1,
                sr_values=[0.5],
                ts_values=[700.0],
                tmeas_values=[300.0],
                substrates=["STO"],
                psyn_mbar_values=[1.333223684e-5],
                pc_mbar_values=[1.333223684e-5],
                annealing_values=[0],
                growth_method="MBE",
                oxygen_activation="No",
                ta_deg_c=350.0,
                pa_mbar=212.275875,
                ta_h=2.0,
                thickness_nm=30.0,
            )
        )
        payload = result.model_dump(mode="json")
        self.assertEqual(payload["candidate_count"], 1)
        self.assertIsNone(payload["all_candidates"][0]["TA"])
        self.assertIsNone(payload["all_candidates"][0]["PA"])
        self.assertIsNone(payload["all_candidates"][0]["tA"])
        self.assertNotIn("Substrate", backend_app.features)

    def test_endpoint_reports_training_range_extrapolation(self) -> None:
        request = backend_app.InverseDesignReq(
            top_n=1,
            sr_values=[0.5],
            ts_values=[2_000.0],
            tmeas_values=[300.0],
            substrates=["STO"],
            psyn_mbar_values=[1.333223684e-5],
            pc_mbar_values=[1.333223684e-5],
            annealing_values=[1],
        )
        result = backend_app.run_inverse_design(request)

        self.assertIn("Ts", result.extrapolated_features)
        self.assertEqual(result.extrapolation_counts["Ts"], 1)

    def test_endpoint_rejects_invalid_request_values(self) -> None:
        invalid_requests = (
            backend_app.InverseDesignReq(target_rho_uohm_cm=float("inf")),
            backend_app.InverseDesignReq(sr_values=[]),
            backend_app.InverseDesignReq(sr_values=[-0.1]),
            backend_app.InverseDesignReq(psyn_mbar_values=[0.0]),
            backend_app.InverseDesignReq(pc_mbar_values=[-1.0]),
            backend_app.InverseDesignReq(pa_mbar=0.0),
            backend_app.InverseDesignReq(annealing_values=[0.5]),
            backend_app.InverseDesignReq(substrates=["Si"]),
            backend_app.InverseDesignReq(growth_method="CVD"),
            backend_app.InverseDesignReq(oxygen_activation="Yes"),
        )
        for request in invalid_requests:
            with self.subTest(request=request):
                with self.assertRaises(HTTPException) as raised:
                    backend_app.run_inverse_design(request)
                self.assertEqual(raised.exception.status_code, 400)

    def test_endpoint_rejects_a_grid_over_the_hard_limit(self) -> None:
        request = backend_app.InverseDesignReq(
            sr_values=[0.5] * (MAX_INVERSE_CANDIDATES + 1),
            ts_values=[700.0],
            tmeas_values=[300.0],
            substrates=["STO"],
            psyn_mbar_values=[1e-5],
            pc_mbar_values=[1e-5],
            annealing_values=[1],
        )
        with self.assertRaises(HTTPException) as raised:
            backend_app.run_inverse_design(request)

        self.assertEqual(raised.exception.status_code, 400)
        self.assertEqual(raised.exception.detail["candidate_limit"], 20_000)


if __name__ == "__main__":
    unittest.main()
