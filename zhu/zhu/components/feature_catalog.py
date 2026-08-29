"""Presentation vocabulary for LSCO data fields.

This module is deliberately *not* a model schema.  The backend ``/features``
response remains the runtime authority for required model inputs.  Pages can
use this catalog to turn a model field name into a Chinese-first label, unit,
and help text without accidentally making the catalog a second validation
source of truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Literal

from .locale import DEFAULT_LANGUAGE


FieldStatus = Literal["input", "output", "not_used", "removed"]
ValueKind = Literal["numeric", "binary", "categorical"]


@dataclass(frozen=True)
class FeatureDefinition:
    """Human-readable metadata for one experimental or model field."""

    key: str
    name_zh: str
    name_en: str
    unit: str
    description_zh: str
    value_kind: ValueKind
    status: FieldStatus
    display_key: str | None = None
    control_label: str | None = None
    options: tuple[str, ...] = ()
    options_en: tuple[str, ...] = ()
    group: str = ""
    note_zh: str = ""
    note_en: str = ""
    api_unit: str | None = None

    @property
    def visible_key(self) -> str:
        """Return the symbol used in a Chinese-first field label."""

        return self.display_key or self.key

    @property
    def label(self) -> str:
        """Return the default (English) compact input label."""

        return self.label_for(DEFAULT_LANGUAGE)

    def label_for(self, language: str = DEFAULT_LANGUAGE) -> str:
        """Return a compact label in the selected UI language."""

        if self.control_label:
            if language != "en":
                return self.control_label
            return f"{self.visible_key} · {self.name_en}"
        unit = f"（{self.unit}）" if self.unit else ""
        english_unit = "dimensionless" if self.unit == "无量纲" else self.unit
        unit_en = f" ({english_unit})" if english_unit else ""
        return (
            f"{self.visible_key} · {self.name_en}{unit_en}"
            if language == "en"
            else f"{self.visible_key} · {self.name_zh}{unit}"
        )

    def workstation_label_for(self, language: str = DEFAULT_LANGUAGE) -> str:
        """Return a short label for the narrow desktop parameter column.

        The full scientific definition remains available in the field
        tooltip. English descriptions are too long for a classic label/input
        row, so the workbench uses the familiar variable symbol and unit.
        """

        if language != "en":
            return self.label_for(language)
        if self.key == "A":
            return "A (0/1)"
        if self.key == "Sr":
            return "Sr (fraction)"
        if self.value_kind == "categorical":
            return self.visible_key
        unit = "" if self.unit == "无量纲" else self.unit
        return f"{self.visible_key} ({unit})" if unit else self.visible_key

    @property
    def bilingual_name(self) -> str:
        """Return the Chinese title followed by its English definition."""

        return f"{self.name_zh} / {self.name_en}"

    @property
    def help_text(self) -> str:
        """Return concise, user-facing explanatory text."""

        parts = [self.bilingual_name, self.description_zh]
        if self.options:
            parts.append("可选值：" + "、".join(self.options))
        if self.note_zh:
            parts.append(self.note_zh)
        return "；".join(part for part in parts if part)

    def help_text_for(self, language: str = DEFAULT_LANGUAGE) -> str:
        if language == "en":
            parts = [self.name_en]
            english_options = self.options_en or self.options
            if english_options:
                parts.append("Options: " + ", ".join(english_options))
            if self.note_en:
                parts.append(self.note_en)
            return "; ".join(part for part in parts if part)
        return self.help_text


# These fields are the supplied GWO-75 Feuil5 model inputs. They intentionally
# exclude ``a`` and ``c``. A page still intersects this catalog with backend
# metadata rather than using this tuple as a validation rule.
ACTIVE_UI_FIELDS: Final[tuple[FeatureDefinition, ...]] = (
    FeatureDefinition(
        key="Psyn",
        name_zh="合成/生长氧压",
        name_en="Synthesis/growth oxygen pressure",
        unit="mbar",
        description_zh="薄膜合成或生长过程中的氧压。",
        value_kind="numeric",
        status="input",
        group="制备条件",
    ),
    FeatureDefinition(
        key="Oxygen activation",
        name_zh="氧活化方式",
        name_en="Oxygen activation method",
        unit="无量纲",
        description_zh="氧活化处理的方法。",
        value_kind="categorical",
        status="input",
        options=("No", "Ozone"),
        group="制备条件",
    ),
    FeatureDefinition(
        key="Mismatch",
        name_zh="晶格失配",
        name_en="Lattice mismatch",
        unit="%",
        description_zh="薄膜与衬底之间的晶格失配。",
        value_kind="numeric",
        status="input",
        group="制备条件",
    ),
    FeatureDefinition(
        key="Ts",
        name_zh="生长/衬底温度",
        name_en="Growth/substrate temperature",
        unit="°C",
        description_zh="薄膜生长或衬底所用温度。",
        value_kind="numeric",
        status="input",
        group="制备条件",
    ),
    FeatureDefinition(
        key="Growth method",
        name_zh="薄膜生长方法",
        name_en="Film growth method",
        unit="无量纲",
        description_zh="制备薄膜所使用的生长方法。",
        value_kind="categorical",
        status="input",
        options=("PLD", "MBE"),
        group="制备条件",
    ),
    FeatureDefinition(
        key="A",
        name_zh="退火状态",
        name_en="Annealing status",
        unit="无量纲",
        description_zh="是否执行退火。",
        value_kind="binary",
        status="input",
        control_label="A · 退火状态（0=未退火，1=已退火）",
        options=("0 = 未退火", "1 = 已退火"),
        options_en=("0 = not annealed", "1 = annealed"),
        group="退火条件",
    ),
    FeatureDefinition(
        key="TA",
        name_zh="退火温度",
        name_en="Annealing temperature",
        unit="°C",
        description_zh="退火过程所用温度。",
        value_kind="numeric",
        status="input",
        group="退火条件",
    ),
    FeatureDefinition(
        key="PA",
        name_zh="退火氧压",
        name_en="Annealing oxygen pressure",
        unit="mbar",
        description_zh="退火过程中的氧压。",
        value_kind="numeric",
        status="input",
        group="退火条件",
    ),
    FeatureDefinition(
        key="tA",
        name_zh="退火时长",
        name_en="Annealing duration",
        unit="h",
        description_zh="退火持续时间。",
        value_kind="numeric",
        status="input",
        group="退火条件",
    ),
    FeatureDefinition(
        key="Pc",
        name_zh="冷却氧压",
        name_en="Cooling oxygen pressure",
        unit="mbar",
        description_zh="冷却阶段的氧压。",
        value_kind="numeric",
        status="input",
        group="冷却与薄膜",
    ),
    FeatureDefinition(
        key="t",
        name_zh="薄膜厚度",
        name_en="Film thickness",
        unit="nm",
        description_zh="LSCO 薄膜的厚度。",
        value_kind="numeric",
        status="input",
        group="冷却与薄膜",
    ),
    FeatureDefinition(
        key="Sr",
        name_zh="锶掺杂分数",
        name_en="Strontium content/doping fraction",
        unit="无量纲",
        description_zh="锶掺杂的摩尔分数，例如 0.05 表示 5%。",
        value_kind="numeric",
        status="input",
        group="成分与测量",
    ),
    FeatureDefinition(
        key="Tmeas",
        name_zh="测量温度",
        name_en="Measurement temperature",
        unit="K",
        description_zh="采集电阻率时的测量温度。",
        value_kind="numeric",
        status="input",
        group="成分与测量",
    ),
)


# Display grouping only: pages may use these names to order a backend-provided
# feature list, but must not treat them as a runtime requirement schema.
FEATURE_GROUPS: Final[dict[str, tuple[str, ...]]] = {
    "制备条件": ("Psyn", "Oxygen activation", "Mismatch", "Ts", "Growth method"),
    "退火条件": ("A", "TA", "PA", "tA"),
    "冷却与薄膜": ("Pc", "t"),
    "成分与测量": ("Sr", "Tmeas"),
}


# This supplied field is not a feature column of the GWO-75 model. It remains
# visible in the reference catalog instead of being silently discarded.
CURRENTLY_UNUSED_FIELDS: Final[tuple[FeatureDefinition, ...]] = (
    FeatureDefinition(
        key="Substrate",
        name_zh="衬底材料/取向",
        name_en="Substrate material/orientation",
        unit="无量纲",
        description_zh="衬底材料及其晶向。",
        value_kind="categorical",
        status="not_used",
        note_zh="当前模型未使用，不会出现在预测表单或批量模板中。",
    ),
)


# ``a`` and ``c`` are retained only as explicit historical vocabulary.  They
# must never be drawn from ``ACTIVE_UI_FIELDS`` for a new form or template.
REMOVED_FIELDS: Final[tuple[FeatureDefinition, ...]] = (
    FeatureDefinition(
        key="a",
        name_zh="晶格常数 a",
        name_en="Lattice parameter a",
        unit="Å",
        description_zh="面内晶格参数。",
        value_kind="numeric",
        status="removed",
        note_zh="已从当前 GWO-75 Feuil5 模型字段移除。",
    ),
    FeatureDefinition(
        key="c",
        name_zh="晶格常数 c",
        name_en="Lattice parameter c",
        unit="Å",
        description_zh="面外晶格参数。",
        value_kind="numeric",
        status="removed",
        note_zh="已从当前 GWO-75 Feuil5 模型字段移除。",
    ),
)


# Feuil5 and the supplied GWO-75 artifact use Ω·cm internally. ``api_unit``
# captures the browser/API conversion independently rather than concealing the
# 1e6 scale gap.
RHO_OUTPUT: Final[FeatureDefinition] = FeatureDefinition(
    key="rho",
    display_key="ρ",
    name_zh="电阻率",
    name_en="Electrical resistivity",
    unit="Ω·cm",
    description_zh="材料的电阻率目标值。",
    value_kind="numeric",
    status="output",
    api_unit="μΩ·cm",
    note_zh="模型内部标签为 Ω·cm；当前 API 与下载结果会转换为 μΩ·cm 交付。",
)


ALL_FIELDS: Final[tuple[FeatureDefinition, ...]] = (
    *ACTIVE_UI_FIELDS,
    *CURRENTLY_UNUSED_FIELDS,
    *REMOVED_FIELDS,
    RHO_OUTPUT,
)
FEATURE_BY_KEY: Final[dict[str, FeatureDefinition]] = {
    definition.key: definition for definition in ALL_FIELDS
}
_KEY_ALIASES: Final[dict[str, str]] = {"ρ": "rho"}


def get_feature_definition(key: str) -> FeatureDefinition | None:
    """Return catalog metadata for ``key`` without changing model behavior."""

    return FEATURE_BY_KEY.get(_KEY_ALIASES.get(str(key), str(key)))


def feature_label(key: str, *, fallback_to_key: bool = True, language: str = DEFAULT_LANGUAGE) -> str:
    """Return a localized control label, or the original key if unknown."""

    definition = get_feature_definition(key)
    if definition is not None:
        return definition.label_for(language)
    return str(key) if fallback_to_key else ""


def feature_help_text(key: str, *, language: str = DEFAULT_LANGUAGE) -> str:
    """Return catalog help text; unknown runtime fields get a safe fallback."""

    definition = get_feature_definition(key)
    if definition is not None:
        return definition.help_text_for(language)
    return (
        "This field has no confirmed display definition; use the same scale as the training data."
        if language == "en"
        else "此字段没有已确认的展示目录说明；请使用与训练数据一致的尺度。"
    )


def get_feature_label(feature: str, *, language: str = DEFAULT_LANGUAGE) -> str:
    """Compatibility-friendly public name for :func:`feature_label`."""

    return feature_label(feature, language=language)


def get_feature_hint(feature: str, *, language: str = DEFAULT_LANGUAGE) -> str:
    """Compatibility-friendly public name for :func:`feature_help_text`."""

    return feature_help_text(feature, language=language)


def get_feature_workstation_label(feature: str, *, language: str = DEFAULT_LANGUAGE) -> str:
    """Return the compact label used by fixed-width workstation forms."""

    definition = get_feature_definition(feature)
    if definition is None:
        return str(feature)
    return definition.workstation_label_for(language)


__all__ = [
    "ACTIVE_UI_FIELDS",
    "ALL_FIELDS",
    "CURRENTLY_UNUSED_FIELDS",
    "FEATURE_BY_KEY",
    "FEATURE_GROUPS",
    "REMOVED_FIELDS",
    "RHO_OUTPUT",
    "FeatureDefinition",
    "FieldStatus",
    "ValueKind",
    "feature_help_text",
    "feature_label",
    "get_feature_hint",
    "get_feature_label",
    "get_feature_workstation_label",
    "get_feature_definition",
]
