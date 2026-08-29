"""Small, dependency-free validation helpers shared by the prediction pages."""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from typing import Any


def validate_required_numeric(
    values: Mapping[str, Any],
    required_fields: Iterable[str],
    binary_fields: Iterable[str] = ("A",),
    category_mappings: Mapping[str, Mapping[str, Any]] | None = None,
    input_transforms: Mapping[str, str] | None = None,
    conditional_missing_when_a_zero: Iterable[str] = (),
) -> dict[str, str]:
    """Return ``{field: Chinese error message}``; an empty mapping means valid.

    Values must be present. Numeric fields must be finite, binary fields must
    be exactly 0 or 1, and fields in ``category_mappings`` must use a declared
    category label (or a declared numeric code). Extra keys are ignored.
    """

    errors: dict[str, str] = {}
    binary = set(binary_fields)
    categories = category_mappings or {}
    transforms = input_transforms or {}
    conditional = set(conditional_missing_when_a_zero)
    try:
        annealing_is_disabled = float(values.get("A", float("nan"))) == 0.0
    except (TypeError, ValueError):
        annealing_is_disabled = False

    for field in required_fields:
        value = values.get(field)
        if value is None or (isinstance(value, str) and not value.strip()):
            errors[field] = "此项为必填项"
            continue

        mapping = categories.get(field)
        if isinstance(mapping, Mapping) and mapping:
            label = str(value).strip()
            if label in mapping:
                continue
            try:
                numeric_code = float(value)
                allowed = {float(code) for code in mapping.values()}
            except (TypeError, ValueError):
                errors[field] = "请选择允许的类别"
                continue
            if not math.isfinite(numeric_code) or numeric_code not in allowed:
                errors[field] = "请选择允许的类别"
            continue

        # ``bool`` is technically an ``int`` in Python but is not an explicit
        # numeric input from the user and should not silently pass validation.
        if isinstance(value, bool):
            errors[field] = "请输入有限数字"
            continue

        try:
            number = float(value)
        except (TypeError, ValueError):
            errors[field] = "请输入有限数字"
            continue

        if not math.isfinite(number):
            errors[field] = "请输入有限数字"
        elif field in binary and number not in (0.0, 1.0):
            errors[field] = "仅允许选择 0 或 1"
        elif (
            transforms.get(field) == "log10_positive"
            and number <= 0.0
            and not (annealing_is_disabled and field in conditional)
        ):
            errors[field] = "请输入大于 0 的数值"

    return errors


def validate_temperature_range(tmin: Any, tmax: Any) -> tuple[bool, str]:
    """Return ``(is_valid, message)`` for a strictly increasing finite range."""

    try:
        lower = float(tmin)
        upper = float(tmax)
    except (TypeError, ValueError):
        return False, "温度范围必须是数字"

    if not math.isfinite(lower) or not math.isfinite(upper):
        return False, "温度范围必须是有限数字"
    if upper <= lower:
        return False, "最高温度必须大于最低温度"
    return True, ""
