"""Figure 19-derived colours for the LSCO scientific workspace.

The document's Figure 19 combines a cool blue-to-warm-orange diverging field
with a plasma-like indigo / purple / yellow accent scale.  These constants
keep that visual vocabulary consistent across the Rio theme and exported
prediction figure while retaining a light, form-friendly workspace.
"""

from __future__ import annotations


FIG19_WHITE = "FFFFFF"
FIG19_INK = "000000"
FIG19_DEEP_BLUE = "0000FF"
FIG19_COOL_BLUE = "96ACEB"
FIG19_COOL_PALE = "D9E1F8"
FIG19_WARM_ORANGE = "FF0000"
FIG19_WARM_PALE = "FCEADE"
FIG19_PURPLE = "8F0DA4"
FIG19_YELLOW = "F0F921"
FIG19_RED = "FF0000"
FIG19_GRAY = "D3D3D3"
FIG19_MUTED_GRAY = "000000"

# Classic desktop scientific-tool palette.  The user-provided reference uses
# neutral system greys, thin borders, and a single restrained blue accent.
# Keep the Figure-19 colours above for the exported scientific plot, while
# these tokens define the application chrome and workbench surfaces.
DESKTOP_BACKGROUND = "F0F0F0"
DESKTOP_PANEL = "F7F7F7"
DESKTOP_SURFACE = "FFFFFF"
DESKTOP_BORDER = "C8C8C8"
DESKTOP_INPUT_BORDER = "9A9A9A"
DESKTOP_INK = "000000"
DESKTOP_MUTED = "000000"
DESKTOP_ACCENT = "0000FF"
DESKTOP_ACCENT_MUTED = "0000FF"
DESKTOP_ACCENT_PALE = "E1EAF4"
DESKTOP_TAB = "E9E9E9"


__all__ = [
    "FIG19_COOL_BLUE",
    "FIG19_COOL_PALE",
    "FIG19_DEEP_BLUE",
    "FIG19_GRAY",
    "FIG19_INK",
    "FIG19_MUTED_GRAY",
    "FIG19_PURPLE",
    "FIG19_RED",
    "FIG19_WARM_ORANGE",
    "FIG19_WARM_PALE",
    "FIG19_WHITE",
    "FIG19_YELLOW",
    "DESKTOP_ACCENT",
    "DESKTOP_ACCENT_MUTED",
    "DESKTOP_ACCENT_PALE",
    "DESKTOP_BACKGROUND",
    "DESKTOP_BORDER",
    "DESKTOP_INK",
    "DESKTOP_INPUT_BORDER",
    "DESKTOP_MUTED",
    "DESKTOP_PANEL",
    "DESKTOP_SURFACE",
    "DESKTOP_TAB",
]
