"""Reusable Rio components for the LSCO scientific workspace.

The module intentionally contains only presentation primitives.  Page-specific
state, API calls, and file parsing stay in their corresponding page modules.
"""

from __future__ import annotations

from dataclasses import field as dataclass_field
from typing import Any, Sequence

import rio
from rio.components.class_container import ClassContainer

from .palette import (
    DESKTOP_ACCENT,
    DESKTOP_ACCENT_PALE,
    DESKTOP_BACKGROUND,
    DESKTOP_BORDER,
    DESKTOP_INK,
    DESKTOP_MUTED,
    DESKTOP_PANEL,
    DESKTOP_SURFACE,
    DESKTOP_TAB,
    FIG19_WHITE,
)
from .locale import LANGUAGE_OPTIONS, language_for, set_language, translate


_BANNER_KINDS: dict[str, str] = {
    "info": "info",
    "success": "success",
    "warning": "warning",
    "danger": "danger",
    "error": "danger",
}

_STATUS_ICONS: dict[str, str] = {
    "info": "material/info",
    "success": "material/check_circle",
    # Rio 0.11's bundled Material registry exposes ``warning`` but not the
    # later ``warning_amber`` alias.  Use the registered name so an
    # extrapolation banner cannot make the entire page fail to build.
    "warning": "material/warning",
    "danger": "material/error",
    "error": "material/error",
}


def primary_navigation_names(component: rio.Component | None = None) -> list[str]:
    """Return the four fixed, desktop-workbench navigation labels."""

    return [
        translate("单次预测", component),
        translate("批量预测", component),
        translate("逆向设计", component),
        translate("模型微调", component),
    ]


def _status_color(component: rio.Component, kind: str) -> rio.Color:
    """Return the semantic colour for a compact status surface."""

    theme = component.session.theme
    normalized = _BANNER_KINDS.get(kind, "info")
    if normalized == "success":
        return theme.success_color
    if normalized == "warning":
        return theme.warning_color
    if normalized == "danger":
        return theme.danger_color
    return theme.primary_color


def _context_chip(text: str, color: rio.Color, *, min_width: float) -> rio.Component:
    """Render a compact desktop-style context field."""

    return rio.Rectangle(
        content=rio.Text(
            text,
            style=rio.TextStyle(font_size=0.84, font_weight="bold", fill=color),
            overflow="ellipsize",
            margin_x=0.5,
            margin_y=0.22,
        ),
        fill=rio.Color.from_hex(DESKTOP_SURFACE),
        stroke_width=0.05,
        stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
        corner_radius=0.06,
        min_width=min_width,
        grow_x=True,
    )


class StatusBanner(rio.Component):
    """A visible semantic status message; accepts ``error`` as an alias."""

    text: str = ""
    kind: str = "info"

    def build(self) -> rio.Component:
        color = _status_color(self, self.kind)
        return rio.Rectangle(
            content=rio.Row(
                rio.Icon(
                    _STATUS_ICONS.get(self.kind, "material/info"),
                    fill=color,
                    min_width=0.72,
                    min_height=0.72,
                ),
                rio.Text(
                    translate(self.text, self),
                    style=rio.TextStyle(font_size=1.0, font_weight="bold"),
                    overflow="ellipsize",
                    grow_x=True,
                ),
                spacing=0.32,
                margin_x=0.5,
                margin_y=0.08,
                # Different Material glyphs have slightly different natural
                # SVG heights (for example info vs. check_circle). Reserve
                # the taller line from the outset so a status change cannot
                # shift every workbench panel below it.
                min_height=1.12,
                grow_x=True,
            ),
            fill=rio.Color.from_hex(DESKTOP_SURFACE),
            stroke_width=0.05,
            stroke_color=color.replace(opacity=0.48),
            corner_radius=0.05,
            min_height=1.62,
            grow_x=True,
        )


class PageShell(rio.Component):
    """Compact workstation frame for one fixed-screen workflow page."""

    title: str
    subtitle: str = ""
    content: rio.Component | None = None
    status_text: str = ""
    status_kind: str = "info"
    model_label: str = ""
    output_unit: str = "μΩ·cm"
    # Most workflow pages deliberately keep their content at its natural
    # height. The prediction workbench is different: its two panels have a
    # stable result skeleton and should occupy the remaining viewport so the
    # chart can use otherwise empty space without creating a scroll region.
    fill_height: bool = False

    def build(self) -> rio.Component:
        title_block = rio.Row(
            rio.Text(
                translate(self.title, self),
                style=rio.TextStyle(font_size=1.0, font_weight="bold"),
                overflow="ellipsize",
                min_width=18,
            ),
            *(
                [
                    rio.Text(
                        translate(self.subtitle, self),
                        style=rio.TextStyle(font_size=0.92, fill=rio.Color.from_hex(DESKTOP_MUTED)),
                        overflow="ellipsize",
                        grow_x=True,
                    )
                ]
                if self.subtitle.strip()
                else []
            ),
            spacing=0.55,
            grow_x=True,
            align_y=0.5,
        )
        context = rio.Row(
            _context_chip(self.model_label or "LSCO model", self.session.theme.primary_color, min_width=17),
            _context_chip(self.output_unit, self.session.theme.secondary_color, min_width=5),
            spacing=0.3,
            align_x=1,
            grow_x=True,
        )
        heading = rio.Rectangle(
            content=rio.Row(
                title_block,
                context,
                proportions=[0.69, 0.31],
                spacing=0.8,
                grow_x=True,
                align_y=0.5,
            ),
            fill=rio.Color.from_hex(DESKTOP_PANEL),
            stroke_width=0.05,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0.05,
            margin_x=0.55,
            margin_y=0.14,
            min_height=1.72,
            grow_x=True,
        )
        body = self.content or rio.Spacer(grow_x=False, grow_y=False)
        return rio.Column(
            heading,
            StatusBanner(translate(self.status_text, self), self.status_kind),
            body,
            spacing=0.18,
            margin=0.22,
            grow_x=True,
            grow_y=True,
            # An explicit vertical alignment wraps the Column in a
            # min-content element in Rio, which discards the space granted by
            # PageView. Only opt out of that wrapper for fixed-screen pages.
            align_y=None if self.fill_height else 0,
        )


class SectionCard(rio.Component):
    """A neutral card with a compact heading and optional supporting text."""

    title: str
    content: rio.Component | None = None
    subtitle: str | None = None
    icon: str | None = None
    accent: rio.Color | None = None
    dense: bool = False
    fill_height: bool = False
    expand_content: bool = False

    def build(self) -> rio.Component:
        header_items: list[rio.Component] = []
        if self.icon:
            header_items.append(
                rio.Icon(
                    self.icon,
                    fill=self.session.theme.primary_color,
                    min_width=0.9,
                    min_height=0.9,
                )
            )
        header_items.append(
            rio.Text(
                translate(self.title, self),
                style=rio.TextStyle(font_size=1.02, font_weight="bold"),
                # Dense workbench cards must retain a stable header height.
                # Supporting details remain available in their field tooltips,
                # rather than making a whole page taller when a label wraps.
                overflow="ellipsize" if self.dense else "wrap",
                grow_x=True,
            )
        )
        card_items: list[rio.Component] = [rio.Row(*header_items, spacing=0.3, min_height=1.12)]
        if self.subtitle:
            card_items.append(
                rio.Text(
                    translate(self.subtitle, self),
                    style="dim",
                    overflow="ellipsize" if self.dense else "wrap",
                )
            )
        if self.content is not None:
            card_items.append(rio.Separator(color=rio.Color.from_hex(DESKTOP_BORDER)))
            card_items.append(self.content)

        card_content = rio.Column(
            *card_items,
            spacing=0.14 if self.dense else 0.4,
            margin=0.3 if self.dense else 0.55,
            grow_x=True,
            grow_y=self.fill_height,
            align_y=None if self.expand_content else (0 if self.fill_height else None),
        )
        return rio.Rectangle(
            content=card_content,
            fill=rio.Color.from_hex(DESKTOP_PANEL),
            stroke_width=0.055,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0.06,
            grow_x=True,
            grow_y=self.fill_height,
        )


class MetricCard(rio.Component):
    """A concise result summary card for one numerical metric."""

    label: str
    value: str
    detail: str | None = None

    def build(self) -> rio.Component:
        items: list[rio.Component] = [
            rio.Text(
                translate(self.label, self),
                style="dim",
                overflow="ellipsize",
                justify="center",
                grow_x=True,
            ),
            rio.Text(
                self.value,
                style="heading3",
                overflow="ellipsize",
                justify="center",
                grow_x=True,
            ),
        ]
        if self.detail:
            items.append(
                rio.Tooltip(
                    anchor=rio.Text(
                        translate(self.detail, self),
                        style="dim",
                        overflow="ellipsize",
                        justify="center",
                        grow_x=True,
                    ),
                    tip=rio.Text(translate(self.detail, self), overflow="wrap"),
                    position="right",
                    grow_x=True,
                )
            )
        return rio.Rectangle(
            content=rio.Column(
                *items,
                spacing=0.12,
                margin_x=0.35,
                margin_y=0.28,
                grow_x=True,
            ),
            fill=rio.Color.from_hex(DESKTOP_SURFACE),
            stroke_width=0.05,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0.04,
            # Four cards must fit inside the 66% result pane at a 1440 px
            # desktop viewport. A larger intrinsic minimum makes Rio widen
            # the entire workbench even though the labels can ellipsize.
            min_width=5.5,
            grow_x=True,
        )


class UploadCard(rio.Component):
    """A labelled drag-and-drop file picker with an optional file summary."""

    title: str = "上传文件"
    description: str = ""
    file_types: tuple[str, ...] = ()
    on_pick_file: rio.EventHandler[rio.FilePickEvent] = None
    multiple: bool = False
    file_summary: str = ""

    def build(self) -> rio.Component:
        picker = rio.FilePickerArea(
            content=translate("点击或拖拽上传", self),
            file_types=list(self.file_types) or None,
            multiple=self.multiple,
            on_pick_file=self.on_pick_file,
            min_height=10,
            grow_x=True,
        )
        items: list[rio.Component] = [picker]
        if self.file_summary.strip():
            items.append(rio.Text(self.file_summary, style="dim", overflow="wrap"))
        return SectionCard(
            self.title,
            rio.Column(*items, spacing=0.7),
            subtitle=self.description or None,
            icon="material/upload_file",
        )


class ScrollablePreviewTable(rio.Component):
    """Compatibility-named, fixed-size preview without a scroll surface.

    The active batch page has richer field/page selectors. This reusable
    primitive intentionally caps its preview as well, so a future page cannot
    accidentally reintroduce horizontal or vertical table scrolling.
    """

    columns: Sequence[str] = ()
    rows: Sequence[Sequence[Any]] = ()
    empty_text: str = "暂无可预览的数据。"
    min_height: float = 16
    max_columns: int = 4
    max_rows: int = 4

    def _table_data(self) -> dict[str, list[str]]:
        visible_columns = list(self.columns)[: self.max_columns]
        data = {str(column): [] for column in visible_columns}
        for row in list(self.rows)[: self.max_rows]:
            for index, column in enumerate(visible_columns):
                value = row[index] if index < len(row) else ""
                data[str(column)].append("" if value is None else str(value))
        return data

    def build(self) -> rio.Component:
        if not self.columns:
            return rio.Text(translate(self.empty_text, self), style="dim", overflow="wrap")

        return rio.Table(
            data=self._table_data(),
            show_row_numbers=False,
            min_width=0,
            min_height=self.min_height,
            grow_x=True,
        )


class FeatureField(rio.Component):
    """A consistently labelled model input; ``is_binary`` renders a 0/1 menu."""

    field: str
    label: str | None = None
    unit_hint: str = "单位待确认"
    text: str = ""
    on_change: rio.EventHandler[Any] = None
    error: str = ""
    is_binary: bool = False
    binary_options: dict[str, str] | None = None
    choice_options: dict[str, str] | None = None
    label_width: float = 12.8
    show_error_text: bool = True

    def build(self) -> rio.Component:
        visible_label = translate(self.label or self.field, self)
        if self.unit_hint.strip():
            visible_label = f"{visible_label}（{self.unit_hint}）"

        if self.is_binary or self.choice_options:
            input_component: rio.Component = rio.Dropdown(
                options=self.choice_options
                or self.binary_options
                or {"请选择": "", "0": "0", "1": "1"},
                label="",
                selected_value=self.text,
                on_change=self.on_change,
                is_valid=not bool(self.error),
                style="rounded",
                grow_x=True,
            )
        else:
            input_component = rio.TextInput(
                label="",
                text=self.text,
                on_change=self.on_change,
                is_valid=not bool(self.error),
                style="rounded",
                grow_x=True,
            )

        row = rio.Row(
            rio.Text(
                visible_label,
                style=rio.TextStyle(font_size=0.92, font_weight="bold", fill=rio.Color.from_hex(DESKTOP_INK)),
                overflow="ellipsize",
                min_width=self.label_width,
                align_y=0.5,
            ),
            input_component,
            spacing=0.35,
            grow_x=True,
            align_y=0.5,
        )
        items: list[rio.Component] = [row]
        if self.error and self.show_error_text:
            items.append(
                rio.Text(
                    translate(self.error, self),
                    fill=self.session.theme.danger_color,
                    overflow="wrap",
                )
            )
        return rio.Column(*items, spacing=0.08, min_width=23, grow_x=True)


class FeatureGroup(rio.Component):
    """Responsive, wrapping group of related feature fields."""

    title: str
    fields: list[rio.Component] = dataclass_field(default_factory=list)
    description: str = ""

    def build(self) -> rio.Component:
        return SectionCard(
            self.title,
            rio.FlowContainer(
                *self.fields,
                spacing=1,
                justify="grow",
                grow_x=True,
            ),
            subtitle=self.description or None,
            icon="material/tune",
        )


class DownloadAction(rio.Component):
    """A browser save action backed by :meth:`rio.Session.save_file`."""

    label: str = "下载结果"
    file_name: str = "download"
    file_contents: bytes | str | None = None
    media_type: str | None = None
    icon: str = "material/download"
    is_sensitive: bool = True
    on_complete: rio.EventHandler[[]] = None
    _error: str = dataclass_field(default="", init=False)

    async def _save(self) -> None:
        if self.file_contents is None:
            self._error = translate("当前没有可下载的结果。", self)
            return
        try:
            await self.session.save_file(
                self.file_contents,
                self.file_name,
                media_type=self.media_type,
            )
        except Exception as exc:  # pragma: no cover - client dependent
            self._error = f"{translate('下载失败：', self)}{exc}"
            return
        self._error = ""
        await self.call_event_handler(self.on_complete)

    def build(self) -> rio.Component:
        items: list[rio.Component] = [
            rio.Button(
                translate(self.label, self),
                icon=self.icon,
                style="minor",
                on_press=self._save,
                is_sensitive=self.is_sensitive and self.file_contents is not None,
            )
        ]
        if self._error:
            items.append(
                rio.Text(
                    self._error,
                    fill=self.session.theme.danger_color,
                    overflow="wrap",
                )
            )
        return rio.Column(*items, spacing=0.3, align_x=0)


class WorkspaceRoot(rio.Component):
    """A compact top-nav shell that replaces Rio's wide default sidebar."""

    @rio.event.on_page_change
    def _on_page_change(self) -> None:
        self.force_refresh()

    def _active_segment(self) -> str:
        """Return the selected route with Rio's root route normalized to ``""``."""

        try:
            return str(self.session.active_page_instances[0].url_segment or "")
        except IndexError:
            return ""

    def _navigate(self, event: rio.SwitcherBarChangeEvent[str]) -> None:
        if event.value is None:
            return
        segment = str(event.value)
        # Rio dispatches the event even for a selected SwitcherBar item in
        # some browser interactions. Navigating to the current route would
        # dirty PageView, rebuild the prediction form, and visibly jump the
        # page back to its initial position. The active view is already shown,
        # so make this a true no-op.
        if segment == self._active_segment():
            return
        self.session.navigate_to("/" if not segment else f"/{segment}")

    def _set_language(self, event: rio.DropdownChangeEvent[str]) -> None:
        """Switch the current browser session without changing the route."""

        set_language(self.session, str(event.value))
        self.force_refresh()
        for page in getattr(self.session, "active_page_instances", ()):
            page.force_refresh()

    def build(self) -> rio.Component:
        active_segment = self._active_segment()
        nav_names = primary_navigation_names(self)
        language = language_for(self)

        language_selector = rio.Dropdown(
            options=LANGUAGE_OPTIONS,
            selected_value=language,
            label=translate("Language", self),
            on_change=self._set_language,
            min_width=10,
            style="rounded",
        )
        application_title = rio.Column(
            rio.Text(
                translate("LSCO Resistivity Prediction", self),
                style=rio.TextStyle(font_size=1.68, font_weight="bold", fill=rio.Color.from_hex(DESKTOP_INK)),
                justify="center",
                overflow="nowrap",
                grow_x=True,
            ),
            rio.Text(
                translate("One application, four workflows: single prediction / batch prediction / inverse design / model fine-tuning", self),
                style=rio.TextStyle(font_size=0.9, font_weight="bold", fill=rio.Color.from_hex(DESKTOP_MUTED)),
                justify="center",
                overflow="ellipsize",
                grow_x=True,
            ),
            spacing=0.16,
            grow_x=True,
        )
        title_band = rio.Rectangle(
            content=rio.Row(
                rio.Spacer(min_width=10),
                application_title,
                language_selector,
                proportions=[0.18, 0.64, 0.18],
                spacing=0.5,
                grow_x=True,
                align_y=0.5,
            ),
            fill=rio.Color.from_hex(DESKTOP_BACKGROUND),
            stroke_width=0.05,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0,
            margin_x=0.55,
            margin_y=0.24,
            min_height=3.9,
            grow_x=True,
        )
        navigation = ClassContainer(
            content=rio.SwitcherBar(
                values=["", "batch", "inverse", "finetune"],
                names=nav_names,
                selected_value=active_segment,
                color="primary",
                on_change=self._navigate,
                grow_x=True,
            ),
            classes=["lsco-primary-navigation"],
            grow_x=True,
        )
        nav_content = rio.Row(
            navigation,
            spacing=0,
            grow_x=True,
            align_y=0.5,
        )

        navbar = rio.Rectangle(
            content=nav_content,
            fill=rio.Color.from_hex(DESKTOP_TAB),
            stroke_width=0.05,
            stroke_color=rio.Color.from_hex(DESKTOP_BORDER),
            corner_radius=0,
            margin_x=0.35,
            margin_y=0.04,
            min_height=2.15,
            grow_x=True,
        )
        return ClassContainer(
            content=rio.Column(
                title_band,
                navbar,
                rio.PageView(grow_x=True, grow_y=True),
                spacing=0,
                grow_x=True,
                grow_y=True,
            ),
            classes=["lsco-desktop-shell"],
            grow_x=True,
            grow_y=True,
        )


__all__ = [
    "DownloadAction",
    "FeatureField",
    "FeatureGroup",
    "MetricCard",
    "PageShell",
    "primary_navigation_names",
    "ScrollablePreviewTable",
    "SectionCard",
    "StatusBanner",
    "UploadCard",
    "WorkspaceRoot",
]
