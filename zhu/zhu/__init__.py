from __future__ import annotations

from pathlib import Path

import rio

from . import components as comps
from .components.palette import (
    DESKTOP_ACCENT,
    DESKTOP_ACCENT_MUTED,
    DESKTOP_ACCENT_PALE,
    DESKTOP_BACKGROUND,
    DESKTOP_BORDER,
    DESKTOP_INK,
    DESKTOP_PANEL,
    DESKTOP_SURFACE,
    DESKTOP_TAB,
    FIG19_COOL_PALE,
    FIG19_COOL_BLUE,
    FIG19_DEEP_BLUE,
    FIG19_INK,
    FIG19_RED,
    FIG19_WARM_ORANGE,
    FIG19_WHITE,
)


# Rio 0.11 does not expose a public page-level overflow policy.  The default
# browser root can therefore acquire a native scrollbar whenever a component
# briefly changes its natural size (for example when a prediction plot is
# populated).  That scrollbar changes the available width and causes a second
# responsive re-layout.  Install this small, idempotent viewport boundary for
# every session before the first component tree is rendered.
_VIEWPORT_LOCK_JAVASCRIPT = """
const styleId = "lsco-fixed-viewport-style";
if (!document.getElementById(styleId)) {
    const style = document.createElement("style");
    style.id = styleId;
    style.textContent = `
        html.lsco-fixed-viewport,
        html.lsco-fixed-viewport body {
            --lsco-workbench-font: "Times New Roman", Tinos, "Liberation Serif", Times, serif;
            --rio-global-font: var(--lsco-workbench-font) !important;
            --rio-global-heading1-font-name: var(--lsco-workbench-font) !important;
            --rio-global-heading2-font-name: var(--lsco-workbench-font) !important;
            --rio-global-heading3-font-name: var(--lsco-workbench-font) !important;
            --rio-global-text-font-name: var(--lsco-workbench-font) !important;
            --rio-global-text-font-size: 1.03rem !important;
            width: 100vw !important;
            height: 100vh !important;
            min-width: 0 !important;
            min-height: 0 !important;
            max-width: 100vw !important;
            max-height: 100vh !important;
            overflow: hidden !important;
            overscroll-behavior: none;
        }

        html.lsco-fixed-viewport body {
            position: fixed;
            inset: 0;
            display: block !important;
            background: #F0F0F0 !important;
            font-family: var(--lsco-workbench-font) !important;
        }

        html.lsco-fixed-viewport .lsco-desktop-shell,
        html.lsco-fixed-viewport .lsco-desktop-shell input,
        html.lsco-fixed-viewport .lsco-desktop-shell button,
        html.lsco-fixed-viewport .lsco-desktop-shell select,
        html.lsco-fixed-viewport .lsco-desktop-shell textarea,
        html.lsco-fixed-viewport .lsco-desktop-shell table {
            font-family: var(--lsco-workbench-font) !important;
        }

        html.lsco-fixed-viewport .lsco-desktop-shell,
        html.lsco-fixed-viewport .lsco-desktop-shell * {
            color: #000000;
        }

        html.lsco-fixed-viewport body > .rio-fundamental-root-component {
            position: absolute;
            inset: 0;
            width: 100% !important;
            height: 100% !important;
            min-width: 0 !important;
            min-height: 0 !important;
            overflow: hidden !important;
            grid-template-columns: minmax(0, 1fr) min-content !important;
        }

        html.lsco-fixed-viewport .rio-user-root-container-outer,
        html.lsco-fixed-viewport .rio-user-root-container-inner,
        html.lsco-fixed-viewport .rio-user-root-container-outer > *,
        html.lsco-fixed-viewport .rio-user-root-container-outer > * > * {
            min-width: 0 !important;
            min-height: 0 !important;
            max-width: 100% !important;
            max-height: 100% !important;
            overflow: hidden !important;
        }

        /* Make the four required top-level routes share the available nav
           row instead of allowing the final tab to be clipped at zoomed
           laptop widths. The selector is scoped to the primary navbar; the
           workflow switchers keep their own natural sizing. */
        .lsco-primary-navigation,
        .lsco-primary-navigation > .rio-switcher-bar {
            min-width: 0 !important;
            max-width: 100% !important;
        }

        .lsco-primary-navigation > .rio-switcher-bar {
            width: 100% !important;
        }

        .lsco-primary-navigation .rio-switcher-bar > div > .rio-switcher-bar-options {
            gap: 1px !important;
            justify-content: stretch !important;
            background: #D8D8D8 !important;
            border-radius: 0 !important;
        }

        .lsco-primary-navigation .rio-switcher-bar > div > .rio-switcher-bar-options > .rio-switcher-bar-option {
            flex: 1 1 0 !important;
            min-width: 0 !important;
            padding: 0.2rem 0.35rem !important;
            border-radius: 0 !important;
        }

        .lsco-primary-navigation .rio-switcher-bar-option,
        .lsco-primary-navigation .rio-switcher-bar-option * {
            font-family: var(--lsco-workbench-font) !important;
            font-size: 1.07rem !important;
            font-weight: 700 !important;
        }

        .lsco-primary-navigation .rio-switcher-bar-marker,
        .lsco-primary-navigation .rio-switcher-bar-marker .rio-switcher-bar-option,
        .lsco-primary-navigation .rio-switcher-bar-marker .rio-switcher-bar-option * {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
        }

        .lsco-primary-navigation .rio-switcher-bar > div > .rio-switcher-bar-options > .rio-switcher-bar-option > div:last-child {
            min-width: 0 !important;
            max-width: 100% !important;
            overflow: hidden !important;
            text-overflow: ellipsis;
        }

        .lsco-parameters-pane .rio-input-box {
            min-height: 1.7rem !important;
            height: 1.7rem !important;
        }

        .lsco-parameters-pane .rio-input-box-padding {
            padding-top: 0.12rem !important;
            padding-bottom: 0.12rem !important;
        }

        .lsco-parameters-pane input,
        .lsco-parameters-pane .rio-dropdown input,
        .lsco-desktop-shell .rio-button .rio-text {
            font-size: 1.04rem !important;
        }

        .lsco-desktop-shell .rio-button .rio-text {
            font-weight: 700 !important;
        }

        .lsco-desktop-shell .rio-button .rio-buttonstyle-major,
        .lsco-desktop-shell .rio-button .rio-buttonstyle-major *,
        .lsco-desktop-shell .rio-button .rio-buttonstyle-major svg,
        .lsco-desktop-shell .rio-button .rio-buttonstyle-major path {
            color: #FFFFFF !important;
            fill: #FFFFFF !important;
            opacity: 1 !important;
        }

        /* Rio's proportional row can fall back to min-content widths after
           a browser font substitution. Keep the four batch summary cells
           visually equal regardless of which Times-compatible face wins. */
        .lsco-summary-strip > .rio-row > div {
            min-width: 0 !important;
            overflow: hidden !important;
        }

        .lsco-summary-strip > .rio-row > div > div {
            display: grid !important;
            grid-template-columns: repeat(4, minmax(0, 1fr)) !important;
            gap: 0.16rem !important;
            width: 100% !important;
            min-width: 0 !important;
        }

        .lsco-summary-strip > .rio-row > div > div > .rio-child-wrapper {
            flex: none !important;
            width: auto !important;
            min-width: 0 !important;
        }

        .lsco-summary-strip > .rio-row > div > div > .rio-child-wrapper > * {
            width: 100% !important;
            min-width: 0 !important;
        }

        /* Fine-tuning controls keep their caption inside the Rio input box.
           They need one extra text line, unlike the external-label parameter
           rows used by single prediction. */
        .lsco-parameters-pane .rio-input-box.has-label {
            min-height: 2.35rem !important;
            height: 2.35rem !important;
        }

        .lsco-parameters-pane .rio-file-picker-area,
        .lsco-parameters-pane .rio-file-picker-area-child-content-container,
        .lsco-parameters-pane .rio-file-picker-area-default-content-container,
        .lsco-parameters-pane .rio-file-picker-area-header,
        .lsco-parameters-pane .rio-file-picker-area-progress {
            min-height: 2.7rem !important;
            height: 2.7rem !important;
        }

        .lsco-parameters-pane .rio-file-picker-area-icon {
            width: 1.7rem !important;
            height: 1.7rem !important;
        }

        .lsco-parameters-pane .rio-file-picker-area-files {
            display: none !important;
        }

        .lsco-parameters-pane .rio-file-picker-area-header {
            box-sizing: border-box !important;
            padding-top: 0.22rem !important;
            padding-bottom: 0.22rem !important;
        }

        .lsco-desktop-shell {
            background: #F0F0F0 !important;
        }

        .lsco-desktop-shell .rio-card {
            box-shadow: none !important;
        }
    `;
    document.head.appendChild(style);
}
document.documentElement.classList.add("lsco-fixed-viewport");
if ("scrollRestoration" in history) {
    history.scrollRestoration = "manual";
}
window.scrollTo(0, 0);
"""


async def _lock_viewport(session: rio.Session) -> None:
    """Install the browser-side boundary that keeps the workbench single-screen."""

    await session._evaluate_javascript(_VIEWPORT_LOCK_JAVASCRIPT)

# Shared visual foundation for the scientific workspace. The colors are drawn
# from the document's Figure 19: deep plasma blue, cool-blue / warm-orange
# divergence, and a red trend/error accent. The workspace stays light so data
# entry remains readable on desktop and narrow screens.
theme = rio.Theme.from_colors(
    primary_color=rio.Color.from_hex(DESKTOP_ACCENT),
    # Keep the everyday interface in the requested white-to-blue family.
    # Figure 19's orange is reserved for semantic warnings rather than being
    # used as a competing decorative accent.
    secondary_color=rio.Color.from_hex(DESKTOP_ACCENT_MUTED),
    background_color=rio.Color.from_hex(DESKTOP_BACKGROUND),
    neutral_color=rio.Color.from_hex(DESKTOP_SURFACE),
    hud_color=rio.Color.from_hex(DESKTOP_PANEL),
    disabled_color=rio.Color.from_hex(DESKTOP_TAB),
    success_color=rio.Color.from_hex("197A54"),
    warning_color=rio.Color.from_hex(FIG19_WARM_ORANGE),
    danger_color=rio.Color.from_hex(FIG19_RED),
    text_color=rio.Color.from_hex(DESKTOP_INK),
    heading_fill="plain",
    corner_radius_small=0.04,
    corner_radius_medium=0.07,
    corner_radius_large=0.1,
    mode="light",
)


# Create the Rio app
app = rio.App(
    build=comps.WorkspaceRoot,
    name="LSCO Resistivity Prediction Workbench",
    theme=theme,
    assets_dir=Path(__file__).parent / "assets",
    on_session_start=_lock_viewport,
)
