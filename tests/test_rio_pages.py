"""Smoke-test every published Rio page without contacting a live API."""

from __future__ import annotations

import asyncio
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import rio
from rio import icon_registry
from rio.components.class_container import ClassContainer
from rio.testing import TestClient


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT / "zhu"))

import zhu  # noqa: E402
from zhu.pages.prediction_page import (  # noqa: E402
    DEFAULT_TEST_VALUES,
    TEMPERATURE_DEFAULT_MAX,
    TEMPERATURE_DEFAULT_MIN,
    PredictionPage,
)
from zhu.components.ui import FeatureField, PageShell, SectionCard, WorkspaceRoot, primary_navigation_names  # noqa: E402


FEATURE_METADATA = {
    "features": [
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
    ],
    "temperature_feature": "Tmeas",
    "training_ranges": {
        feature: {"min": 0.0, "max": 1.0}
        for feature in [
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
        ]
    },
    "input_unit_note": "Input values must match the training-data scale.",
    "target_unit": "μΩ·cm",
    "target_log_unit": "log10(μΩ·cm)",
    "output_transform": "y = 10 ** y_log",
    "model_label": "test-model.joblib",
    "categorical_features": {
        "Oxygen activation": {"No": 0.0, "Ozone": 1.0},
        "Growth method": {"MBE": 0.0, "PLD": 1.0},
    },
    "input_transforms": {
        "Psyn": "log10_positive",
        "PA": "log10_positive",
        "Pc": "log10_positive",
    },
    "conditional_missing_when_a_zero": ["TA", "PA", "tA"],
    "source_target_unit": "ohm_cm",
    "output_log_offset": 6.0,
    "warning": None,
}


async def _run_in_test_loop(function, *args, **kwargs):
    """Deterministic stand-in for ``asyncio.to_thread`` in page smoke tests."""
    return function(*args, **kwargs)


class RioPagesSmokeTest(unittest.IsolatedAsyncioTestCase):
    def test_status_banner_icons_are_registered(self) -> None:
        """Every status state must remain renderable in the pinned Rio build."""

        from zhu.components.ui import _STATUS_ICONS

        for kind, icon_name in _STATUS_ICONS.items():
            with self.subTest(kind=kind, icon=icon_name):
                self.assertTrue(icon_registry.get_icon_svg(icon_name))

    async def test_every_published_page_builds_without_crashing(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = FEATURE_METADATA

        with (
            patch("requests.get", return_value=response),
            patch("zhu.pages.prediction_page.asyncio.to_thread", new=_run_in_test_loop),
            patch("zhu.pages.batch_prediction.asyncio.to_thread", new=_run_in_test_loop),
        ):
            for url in ("/", "/batch", "/inverse", "/finetune"):
                async with TestClient(zhu.app, active_url=url) as client:
                    # Page metadata is fetched by an asynchronous Rio lifecycle
                    # handler. Let the scheduled task complete before closing
                    # the test session, otherwise Python rightfully reports an
                    # unawaited coroutine during teardown.
                    await asyncio.sleep(0)
                    await client.refresh()
                    self.assertEqual(dict(client.crashed_build_functions), {}, url)

            async with TestClient(zhu.app, active_url="/") as client:
                await asyncio.sleep(0)
                page = client.get_component(PredictionPage)
                page.xs = [3.0, 174.0, 345.0]
                page.ys = [1200.0, 1600.0, 2100.0]
                page.last_latency_ms = 8
                page.y_unit = "μΩ·cm"
                figure = page._make_figure()
                try:
                    axis = figure.axes[0]
                    self.assertEqual(len(axis.lines), 0, "prediction points must not be connected by a line")
                    self.assertEqual(len(axis.collections), 1)
                    self.assertEqual(len(axis.collections[0].get_offsets()), 3)
                finally:
                    figure.clear()
                page.curve_preview_png = page._render_curve_png()
                await client.refresh()
                self.assertEqual(dict(client.crashed_build_functions), {}, "prediction result")
                self.assertEqual(list(client.get_components(rio.Plot)), [], "prediction preview must not impose SVG size")

                # An extrapolation uses the warning banner. This was the
                # production crash path, so exercise the real page build as
                # well as checking the icon registry above.
                page.status_text = "Tmeas 超出训练范围，结果属于外推。"
                page.status_kind = "warning"
                await client.refresh()
                self.assertEqual(dict(client.crashed_build_functions), {}, "prediction warning")

                # Full-form validation must keep the fixed workstation height:
                # invalid borders and tooltip details replace per-field rows.
                page.xs = []
                page.ys = []
                page.curve_preview_png = None
                page.field_errors = {
                    feature: "此项为必填项"
                    for feature in page._input_features()
                }
                page.temperature_error = "最高温度必须大于最低温度"
                await client.refresh()
                self.assertEqual(dict(client.crashed_build_functions), {}, "prediction validation")
                errored_fields = [field for field in client.get_components(FeatureField) if field.error]
                self.assertTrue(errored_fields)
                self.assertTrue(all(not field.show_error_text for field in errored_fields))

    async def test_single_prediction_loads_defaults_once_and_preserves_user_edits(self) -> None:
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = FEATURE_METADATA

        with (
            patch("requests.get", return_value=response),
            patch("zhu.pages.prediction_page.asyncio.to_thread", new=_run_in_test_loop),
        ):
            async with TestClient(zhu.app, active_url="/") as client:
                await asyncio.sleep(0)
                page = client.get_component(PredictionPage)
                self.assertEqual(page.values, DEFAULT_TEST_VALUES)
                self.assertEqual(page.tmin, TEMPERATURE_DEFAULT_MIN)
                self.assertEqual(page.tmax, TEMPERATURE_DEFAULT_MAX)

                edited = dict(page.values)
                edited["Psyn"] = "9.5e-5"
                edited["Mismatch"] = ""
                page.values = edited
                page.tmin = "42"
                page.tmax = "84"
                await page._load_metadata()

                self.assertEqual(page.values["Psyn"], "9.5e-5")
                self.assertEqual(page.values["Mismatch"], "")
                self.assertEqual(page.tmin, "42")
                self.assertEqual(page.tmax, "84")

    def test_all_four_workflows_are_published_in_navigation_order(self) -> None:
        self.assertEqual(
            [(page.name, page.url_segment) for page in zhu.app.pages],
            [
                ("Single Prediction", ""),
                ("Batch Prediction", "batch"),
                ("Inverse Design", "inverse"),
                ("Model Fine-tuning", "finetune"),
            ],
        )

    async def test_reselecting_single_prediction_does_not_navigate_or_rebuild(self) -> None:
        """Clicking the selected top-nav item must not cause a visible page jump."""

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = FEATURE_METADATA

        with (
            patch("requests.get", return_value=response),
            patch("zhu.pages.prediction_page.asyncio.to_thread", new=_run_in_test_loop),
        ):
            async with TestClient(zhu.app, active_url="/") as client:
                await asyncio.sleep(0)
                root = client.get_component(WorkspaceRoot)
                with patch.object(root.session, "navigate_to") as navigate_to:
                    root._navigate(rio.SwitcherBarChangeEvent(value=""))
                    navigate_to.assert_not_called()

                    root._navigate(rio.SwitcherBarChangeEvent(value="batch"))
                    navigate_to.assert_called_once_with("/batch")

                    navigate_to.reset_mock()
                    root._navigate(rio.SwitcherBarChangeEvent(value="inverse"))
                    navigate_to.assert_called_once_with("/inverse")

    def test_primary_navigation_uses_fixed_desktop_labels(self) -> None:
        """The workstation always presents its four full workflow names."""

        self.assertEqual(
            primary_navigation_names(),
            ["Single Prediction", "Batch Prediction", "Inverse Design", "Model Fine-tuning"],
        )

        chinese_session = type("Session", (), {"_lsco_language": "zh"})()
        self.assertEqual(primary_navigation_names(chinese_session), ["单次预测", "批量预测", "逆向设计", "模型微调"])

    async def test_prediction_uses_one_fixed_desktop_workspace(self) -> None:
        """The prediction page has no internal narrow-screen switchers."""

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = FEATURE_METADATA

        with (
            patch("requests.get", return_value=response),
            patch("zhu.pages.prediction_page.asyncio.to_thread", new=_run_in_test_loop),
        ):
            async with TestClient(zhu.app, active_url="/") as client:
                # Mobile layouts are out of scope for this local scientific
                # workbench. Even at a constrained width, keep the desktop
                # row/grid rather than replacing panels with extra choices.
                client.session.window_width = 50
                await asyncio.sleep(0)
                await client.refresh()
                self.assertEqual(len(list(client.get_components(rio.SwitcherBar))), 1)
                input_grids = [
                    grid
                    for grid in list(client.get_components(rio.Grid))
                    if len(grid._children) == len(FEATURE_METADATA["features"]) - 1
                ]
                self.assertEqual(len(input_grids), 1)
                semantic_classes = {
                    css_class
                    for container in client.get_components(ClassContainer)
                    for css_class in container.classes
                }
                self.assertTrue(
                    {"lsco-desktop-shell", "lsco-workbench-split", "lsco-parameters-pane", "lsco-results-pane", "lsco-summary-strip", "lsco-plot-region"}
                    .issubset(semantic_classes)
                )
                self.assertEqual(len(list(client.get_components(FeatureField))), len(FEATURE_METADATA["features"]) + 1)
                shell = client.get_component(PageShell)
                self.assertTrue(shell.fill_height)
                self.assertTrue(shell.grow_y)
                main_sections = {
                    section.title: section
                    for section in client.get_components(SectionCard)
                    if section.title in {"Model Inputs", "Resistivity–Temperature Curve"}
                }
                self.assertTrue(main_sections["Model Inputs"].fill_height)
                self.assertFalse(main_sections["Model Inputs"].expand_content)
                self.assertTrue(main_sections["Resistivity–Temperature Curve"].fill_height)
                self.assertTrue(main_sections["Resistivity–Temperature Curve"].expand_content)
                self.assertEqual(
                    {section.title for section in client.get_components(SectionCard)},
                    {"Model Inputs", "Resistivity–Temperature Curve"},
                )

    async def test_desktop_workbenches_keep_all_stages_visible_at_narrow_width(self) -> None:
        """Batch, inverse design, and fine-tune use one desktop layout at every width."""

        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = FEATURE_METADATA

        with (
            patch("requests.get", return_value=response),
            patch("zhu.pages.prediction_page.asyncio.to_thread", new=_run_in_test_loop),
            patch("zhu.pages.batch_prediction.asyncio.to_thread", new=_run_in_test_loop),
        ):
            for url in ("/batch", "/inverse", "/finetune"):
                async with TestClient(zhu.app, active_url=url) as client:
                    # Rio reports dimensions in font-height units. Even at a
                    # small value these desktop-only pages must not replace
                    # their workflow cards with a secondary SwitcherBar.
                    client.session.window_width = 70
                    client.session.window_height = 40
                    await asyncio.sleep(0)
                    await client.refresh()
                    self.assertEqual(dict(client.crashed_build_functions), {}, url)
                    self.assertEqual(list(client.get_components(rio.ScrollContainer)), [], url)
                    semantic_classes = {
                        css_class
                        for container in client.get_components(ClassContainer)
                        for css_class in container.classes
                    }
                    self.assertTrue(
                        {"lsco-workbench-split", "lsco-parameters-pane", "lsco-results-pane"}
                        .issubset(semantic_classes),
                        url,
                    )
                    self.assertGreaterEqual(len(list(client.get_components(SectionCard))), 4, url)
                    shell = client.get_component(PageShell)
                    self.assertTrue(shell.fill_height, url)
                    self.assertTrue(shell.grow_y, url)


if __name__ == "__main__":
    unittest.main()
