"""Focused validation and Rio workflow tests for the inverse-design page."""

from __future__ import annotations

import asyncio
from io import BytesIO
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import Mock, patch

import pandas as pd
from rio.testing import TestClient


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "zhu"))

import zhu  # noqa: E402
from zhu.pages.inverse_design_page import (  # noqa: E402
    MAX_CANDIDATES,
    InverseDesignPage,
    validate_inverse_form,
)


DEFAULT_FORM = {
    "target_rho": "500",
    "top_n": "20",
    "growth_method": "MBE",
    "oxygen_activation": "No",
    "annealing_values": "1",
    "ta_deg_c": "350",
    "pa_mbar": "212.275875",
    "ta_h": "2",
    "thickness_nm": "30",
    "sr_values": "0.0675,0.125,0.25,0.50",
    "ts_values": "600,650,700,750",
    "tmeas_values": "300",
    "substrates": "YSZ,LAO,LSAT,STO,MgO",
    "psyn_mbar_values": "1.333223684e-5,0.0001333223684,0.001333223684,0.01333223684,0.2666447368",
    "pc_mbar_values": "1.333223684e-5,0.0001333223684,0.001333223684,0.01333223684,0.2666447368",
}


def _form_with(**changes: str) -> dict[str, str]:
    form = dict(DEFAULT_FORM)
    form.update(changes)
    return form


async def _run_in_test_loop(function, *args, **kwargs):
    """Run synchronous page helpers deterministically inside Rio tests."""

    return function(*args, **kwargs)


class InverseFormValidationTest(unittest.TestCase):
    def test_default_form_builds_2000_candidate_microohm_request(self) -> None:
        payload, errors, candidate_count = validate_inverse_form(DEFAULT_FORM)

        self.assertEqual(errors, {})
        self.assertEqual(candidate_count, 2_000)
        self.assertIsNotNone(payload)
        assert payload is not None
        self.assertEqual(payload["target_rho_uohm_cm"], 500.0)
        self.assertNotIn("target_rho_ohm_cm", payload)
        self.assertEqual(payload["top_n"], 20)
        self.assertEqual(payload["annealing_values"], [1])
        self.assertEqual(payload["psyn_mbar_values"][0], 1.333223684e-5)

    def test_empty_required_values_are_rejected(self) -> None:
        for field in ("target_rho", "sr_values", "substrates"):
            with self.subTest(field=field):
                payload, errors, _ = validate_inverse_form(_form_with(**{field: ""}))
                self.assertIsNone(payload)
                self.assertIn(field, errors)

    def test_non_finite_values_are_rejected(self) -> None:
        cases = {
            "target_rho": "nan",
            "ts_values": "600,inf",
            "ta_deg_c": "-inf",
        }
        for field, value in cases.items():
            with self.subTest(field=field, value=value):
                payload, errors, _ = validate_inverse_form(_form_with(**{field: value}))
                self.assertIsNone(payload)
                self.assertIn(field, errors)
                self.assertIn("finite", errors[field])

    def test_non_positive_pressures_are_rejected(self) -> None:
        cases = {
            "psyn_mbar_values": "0,0.1",
            "pc_mbar_values": "-0.1,0.1",
            "pa_mbar": "0",
        }
        for field, value in cases.items():
            with self.subTest(field=field, value=value):
                payload, errors, _ = validate_inverse_form(_form_with(**{field: value}))
                self.assertIsNone(payload)
                self.assertIn(field, errors)
                self.assertIn("greater than 0", errors[field])

    def test_sr_outside_mismatch_table_domain_is_rejected(self) -> None:
        for value in ("-0.0001", "0.8001"):
            with self.subTest(value=value):
                payload, errors, _ = validate_inverse_form(_form_with(sr_values=value))
                self.assertIsNone(payload)
                self.assertIn("sr_values", errors)

    def test_annealing_values_must_be_binary(self) -> None:
        for value in ("2", "0.5", "0,1,2"):
            with self.subTest(value=value):
                payload, errors, _ = validate_inverse_form(_form_with(annealing_values=value))
                self.assertIsNone(payload)
                self.assertIn("annealing_values", errors)
                self.assertIn("only 0 and/or 1", errors["annealing_values"])

    def test_top_n_accepts_only_integers_from_1_to_100(self) -> None:
        for value in ("0", "101", "1.5"):
            with self.subTest(value=value):
                payload, errors, _ = validate_inverse_form(_form_with(top_n=value))
                self.assertIsNone(payload)
                self.assertIn("top_n", errors)

        for value in ("1", "100"):
            with self.subTest(valid=value):
                payload, errors, candidate_count = validate_inverse_form(_form_with(top_n=value))
                self.assertIsNotNone(payload)
                self.assertEqual(errors, {})
                self.assertEqual(candidate_count, 2_000)

    def test_candidate_grid_allows_20000_and_rejects_larger_grids(self) -> None:
        exactly_ten_temperatures = ",".join(str(value) for value in range(10))
        payload, errors, candidate_count = validate_inverse_form(
            _form_with(tmeas_values=exactly_ten_temperatures)
        )
        self.assertIsNotNone(payload)
        self.assertEqual(errors, {})
        self.assertEqual(candidate_count, MAX_CANDIDATES)

        eleven_temperatures = ",".join(str(value) for value in range(11))
        payload, errors, candidate_count = validate_inverse_form(
            _form_with(tmeas_values=eleven_temperatures)
        )
        self.assertIsNone(payload)
        self.assertEqual(candidate_count, 22_000)
        self.assertIn("candidate_count", errors)
        self.assertIn("20,000", errors["candidate_count"])


class InverseDesignPageWorkflowTest(unittest.IsolatedAsyncioTestCase):
    async def test_search_builds_in_memory_results_and_download_payloads(self) -> None:
        metadata_response = Mock()
        metadata_response.raise_for_status.return_value = None
        metadata_response.json.return_value = {
            "model_label": "GWO-75",
            "target_unit": "μΩ·cm",
            "warning": None,
        }

        best_row = {
            "Rank": 1,
            "Predicted_rho_uohm_cm": 853.145532,
            "Target_rho_uohm_cm": 500.0,
            "Abs_relative_error_percent": 70.6291064,
            "Abs_error_log10": 0.2320531165,
            "Sr": 0.5,
            "Substrate": "LAO",
            "Mismatch": -1.07,
            "Ts": 700.0,
            "Psyn": 0.2666447368,
            "Pc": 0.01333223684,
            "A": 1,
            "TA": 350.0,
            "PA": 212.275875,
            "tA": 2.0,
            "t": 30.0,
            "Tmeas": 300.0,
            "Oxygen activation": "No",
            "Growth method": "MBE",
            "Extrapolated_features": "TA, PA, tA",
        }
        second_row = dict(best_row)
        second_row.update(
            {
                "Rank": 2,
                "Predicted_rho_uohm_cm": 900.0,
                "Abs_relative_error_percent": 80.0,
                "Substrate": "STO",
                "Mismatch": 1.96,
            }
        )
        search_response = Mock()
        search_response.raise_for_status.return_value = None
        search_response.json.return_value = {
            "model_label": "GWO-75",
            "target_unit": "μΩ·cm",
            "candidate_count": 2_000,
            "top_candidates": [best_row],
            "all_candidates": [best_row, second_row],
            "mismatch_table": [
                {"Sr_fraction": 0.5, "LAO": -1.07, "STO": 1.96}
            ],
            "normalized_config": {
                "target_rho_uohm_cm": 500.0,
                "top_n": 20,
                "candidate_count": 2_000,
            },
            "extrapolated_features": ["TA", "PA", "tA"],
            "latency_ms": 12,
        }

        original_to_csv = pd.DataFrame.to_csv
        original_excel_writer = pd.ExcelWriter
        excel_destinations: list[object] = []

        def to_csv_spy(frame: pd.DataFrame, *args, **kwargs):
            return original_to_csv(frame, *args, **kwargs)

        def excel_writer_spy(destination, *args, **kwargs):
            excel_destinations.append(destination)
            return original_excel_writer(destination, *args, **kwargs)

        with (
            patch(
                "zhu.pages.inverse_design_page.requests.get",
                return_value=metadata_response,
            ) as get_mock,
            patch(
                "zhu.pages.inverse_design_page.requests.post",
                return_value=search_response,
            ) as post_mock,
            patch(
                "zhu.pages.inverse_design_page.asyncio.to_thread",
                new=_run_in_test_loop,
            ),
            patch.object(
                pd.DataFrame,
                "to_csv",
                autospec=True,
                side_effect=to_csv_spy,
            ) as to_csv_mock,
            patch(
                "zhu.pages.inverse_design_page.pd.ExcelWriter",
                side_effect=excel_writer_spy,
            ),
        ):
            async with TestClient(zhu.app, active_url="/inverse") as client:
                await asyncio.sleep(0)
                await client.refresh()
                page = client.get_component(InverseDesignPage)
                self.assertTrue(page.metadata_loaded)
                self.assertEqual(dict(client.crashed_build_functions), {})

                await page._run_search()
                await client.refresh()

                self.assertEqual(dict(client.crashed_build_functions), {})
                self.assertTrue(page.has_results)
                self.assertFalse(page.is_running)
                self.assertEqual(page.candidate_count, 2_000)
                self.assertEqual(page.top_result_count, 1)
                self.assertIsInstance(page._top_results, pd.DataFrame)
                self.assertIsInstance(page._all_results, pd.DataFrame)
                assert page._top_results is not None
                assert page._all_results is not None
                self.assertEqual(len(page._top_results), 1)
                self.assertEqual(len(page._all_results), 2)
                self.assertAlmostEqual(
                    float(page._top_results.iloc[0]["Predicted_rho_uohm_cm"]),
                    853.145532,
                )
                self.assertEqual(page.best_prediction, "853.146")
                self.assertEqual(page.extrapolated_features, ["TA", "PA", "tA"])

                self.assertIsInstance(page._result_csv_bytes, bytes)
                self.assertIsInstance(page._result_xlsx_bytes, bytes)
                self.assertIsInstance(page._config_json_bytes, bytes)
                self.assertTrue(page._result_csv_bytes)
                self.assertTrue(page._result_xlsx_bytes)
                self.assertTrue(page._config_json_bytes)

                csv_frame = pd.read_csv(BytesIO(page._result_csv_bytes))
                self.assertEqual(len(csv_frame), 2)
                self.assertAlmostEqual(
                    float(csv_frame.iloc[0]["Predicted_rho_uohm_cm"]),
                    853.145532,
                )
                workbook = pd.ExcelFile(BytesIO(page._result_xlsx_bytes))
                self.assertEqual(
                    workbook.sheet_names,
                    ["Top candidates", "All candidates", "Mismatch table"],
                )
                config = json.loads(page._config_json_bytes.decode("utf-8"))
                self.assertEqual(config["target_rho_uohm_cm"], 500.0)
                self.assertNotIn("output_dir", config)
                self.assertNotIn("model", config)

                get_mock.assert_called_once()
                post_mock.assert_called_once()
                request_kwargs = post_mock.call_args.kwargs
                self.assertEqual(request_kwargs["json"]["target_rho_uohm_cm"], 500.0)
                self.assertEqual(request_kwargs["json"]["top_n"], 20)
                self.assertNotIn("output_dir", request_kwargs["json"])

                # CSV serialization has no path argument and XLSX receives a
                # BytesIO stream. This guards against regressing to the
                # handoff script's fixed server-side output directory.
                to_csv_mock.assert_called_once()
                csv_args, csv_kwargs = to_csv_mock.call_args
                self.assertEqual(len(csv_args), 1)
                self.assertNotIn("path_or_buf", csv_kwargs)
                self.assertTrue(excel_destinations)
                self.assertTrue(
                    all(isinstance(destination, BytesIO) for destination in excel_destinations)
                )


if __name__ == "__main__":
    unittest.main()
