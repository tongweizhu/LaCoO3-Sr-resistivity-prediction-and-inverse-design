"""Pure helpers for the constrained LSCO inverse-design search.

The fitted GWO-75 estimator remains a forward model.  These helpers construct
an explicit, bounded grid of candidate synthesis conditions and rank the model
predictions by their distance from a target resistivity.  Model loading and
prediction deliberately stay in :mod:`backend_app`, so this module is safe to
test without deserializing a model or writing result files.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import math
from typing import Any, Sequence

import numpy as np
import pandas as pd


TORR_TO_MBAR = 1.333223684
AIR_O2_MBAR = 0.2095 * 1013.25
MAX_INVERSE_CANDIDATES = 20_000
VALID_SUBSTRATES: tuple[str, ...] = ("YSZ", "LAO", "LSAT", "STO", "MgO")
VALID_GROWTH_METHODS: tuple[str, ...] = ("MBE", "PLD")
VALID_OXYGEN_ACTIVATIONS: tuple[str, ...] = ("No", "Ozone")
INVERSE_DESIGN_NOTE = (
    "Candidates are ranked within the supplied discrete search space. This is "
    "not a unique mathematical inverse solution, and every candidate requires "
    "experimental validation."
)


MISMATCH_SR: tuple[float, ...] = (
    0.000,
    0.050,
    0.065,
    0.080,
    0.100,
    0.110,
    0.135,
    0.200,
    0.300,
    0.400,
    0.500,
    0.600,
    0.800,
)

MISMATCH_VALUES: dict[str, tuple[float, ...]] = {
    "LAO": (-0.81, -0.84, -0.85, -0.85, -0.86, -0.87, -0.88, -0.92, -0.97, -1.02, -1.07, -1.12, -1.23),
    "STO": (2.23, 2.20, 2.19, 2.18, 2.17, 2.17, 2.15, 2.12, 2.06, 2.01, 1.96, 1.91, 1.80),
    "LSAT": (1.26, 1.23, 1.22, 1.21, 1.20, 1.20, 1.19, 1.15, 1.10, 1.04, 0.99, 0.94, 0.83),
    "MgO": (10.26, 10.23, 10.22, 10.22, 10.20, 10.20, 10.18, 10.15, 10.09, 10.03, 9.974, 9.916, 9.80),
    "YSZ": (-4.660, -4.685, -4.692, -4.700, -4.710, -4.715, -4.727, -4.759, -4.809, -4.859, -4.909, -4.958, -5.057),
}


def _default_pressure_values() -> tuple[float, ...]:
    return tuple(
        torr * TORR_TO_MBAR for torr in (1e-5, 1e-4, 1e-3, 1e-2, 0.2)
    )


@dataclass(frozen=True)
class InverseDesignConfig:
    """Normalized public-unit request for one bounded grid search."""

    target_rho_uohm_cm: float = 500.0
    top_n: int = 20
    sr_values: tuple[float, ...] = (0.0675, 0.125, 0.25, 0.50)
    ts_values: tuple[float, ...] = (600.0, 650.0, 700.0, 750.0)
    tmeas_values: tuple[float, ...] = (300.0,)
    substrates: tuple[str, ...] = VALID_SUBSTRATES
    psyn_mbar_values: tuple[float, ...] = field(default_factory=_default_pressure_values)
    pc_mbar_values: tuple[float, ...] = field(default_factory=_default_pressure_values)
    growth_method: str = "MBE"
    oxygen_activation: str = "No"
    annealing_values: tuple[int, ...] = (1,)
    ta_deg_c: float = 350.0
    pa_mbar: float = AIR_O2_MBAR
    ta_h: float = 2.0
    thickness_nm: float = 30.0


def _finite_number(value: object, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"{label} must be a finite number")
    return numeric


def _finite_sequence(values: Sequence[object], label: str) -> list[float]:
    if isinstance(values, (str, bytes)) or len(values) == 0:
        raise ValueError(f"{label} must be a non-empty list")
    return [_finite_number(value, f"{label}[{index}]") for index, value in enumerate(values)]


def _positive_integer(value: object, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if not math.isfinite(numeric) or not numeric.is_integer() or numeric <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return int(numeric)


def mismatch_from_sr_substrate(sr_fraction: float, substrate: str) -> float:
    """Interpolate mismatch (%) without silently clamping outside 0..0.8."""

    sr_value = _finite_number(sr_fraction, "Sr")
    if not 0.0 <= sr_value <= 0.8:
        raise ValueError("Sr values must be within the mismatch table domain 0..0.8")
    normalized_substrate = str(substrate).strip()
    if normalized_substrate not in VALID_SUBSTRATES:
        raise ValueError(
            f"Unknown substrate '{normalized_substrate}'. Valid substrates: "
            + ", ".join(VALID_SUBSTRATES)
        )
    return float(
        np.interp(
            sr_value,
            np.asarray(MISMATCH_SR, dtype=float),
            np.asarray(MISMATCH_VALUES[normalized_substrate], dtype=float),
        )
    )


def validate_inverse_config(config: InverseDesignConfig) -> int:
    """Validate a search configuration and return its candidate count."""

    target = _finite_number(config.target_rho_uohm_cm, "target_rho_uohm_cm")
    if target <= 0.0:
        raise ValueError("target_rho_uohm_cm must be positive")
    top_n = _positive_integer(config.top_n, "top_n")
    if top_n > 100:
        raise ValueError("top_n must not exceed 100")

    sr_values = _finite_sequence(config.sr_values, "sr_values")
    if any(value < 0.0 or value > 0.8 for value in sr_values):
        raise ValueError("Sr values must be within the mismatch table domain 0..0.8")
    _finite_sequence(config.ts_values, "ts_values")
    tmeas_values = _finite_sequence(config.tmeas_values, "tmeas_values")
    if any(value <= 0.0 for value in tmeas_values):
        raise ValueError("Tmeas values in K must be positive")

    if isinstance(config.substrates, (str, bytes)) or len(config.substrates) == 0:
        raise ValueError("substrates must be a non-empty list")
    normalized_substrates = [str(value).strip() for value in config.substrates]
    invalid_substrates = [value for value in normalized_substrates if value not in VALID_SUBSTRATES]
    if invalid_substrates:
        raise ValueError(
            "Unknown substrate(s): "
            + ", ".join(invalid_substrates)
            + ". Valid substrates: "
            + ", ".join(VALID_SUBSTRATES)
        )

    psyn_values = _finite_sequence(config.psyn_mbar_values, "psyn_mbar_values")
    pc_values = _finite_sequence(config.pc_mbar_values, "pc_mbar_values")
    if any(value <= 0.0 for value in psyn_values):
        raise ValueError("psyn_mbar values must be positive")
    if any(value <= 0.0 for value in pc_values):
        raise ValueError("pc_mbar values must be positive")

    if isinstance(config.annealing_values, (str, bytes)) or len(config.annealing_values) == 0:
        raise ValueError("annealing_values must be a non-empty list")
    annealing_values: list[int] = []
    for index, value in enumerate(config.annealing_values):
        numeric = _finite_number(value, f"annealing_values[{index}]")
        if not numeric.is_integer() or int(numeric) not in (0, 1):
            raise ValueError("A values must be exactly 0 or 1")
        annealing_values.append(int(numeric))

    growth_method = str(config.growth_method).strip()
    if growth_method not in VALID_GROWTH_METHODS:
        raise ValueError(
            "growth_method must be one of: " + ", ".join(VALID_GROWTH_METHODS)
        )
    oxygen_activation = str(config.oxygen_activation).strip()
    if oxygen_activation not in VALID_OXYGEN_ACTIVATIONS:
        raise ValueError(
            "oxygen_activation must be one of: "
            + ", ".join(VALID_OXYGEN_ACTIVATIONS)
        )

    _finite_number(config.ta_deg_c, "ta_deg_c")
    pa_value = _finite_number(config.pa_mbar, "pa_mbar")
    if pa_value <= 0.0:
        raise ValueError("pa_mbar must be positive")
    ta_h = _finite_number(config.ta_h, "ta_h")
    if ta_h < 0.0:
        raise ValueError("ta_h must be non-negative")
    thickness = _finite_number(config.thickness_nm, "thickness_nm")
    if thickness <= 0.0:
        raise ValueError("thickness_nm must be positive")

    candidate_count = math.prod(
        (
            len(sr_values),
            len(config.ts_values),
            len(config.tmeas_values),
            len(normalized_substrates),
            len(psyn_values),
            len(pc_values),
            len(annealing_values),
        )
    )
    if candidate_count > MAX_INVERSE_CANDIDATES:
        raise ValueError(
            f"Candidate grid contains {candidate_count} rows; maximum is "
            f"{MAX_INVERSE_CANDIDATES}"
        )
    return candidate_count


def build_candidate_grid(config: InverseDesignConfig) -> pd.DataFrame:
    """Build a validated candidate grid using canonical API feature names."""

    expected_count = validate_inverse_config(config)
    rows: list[dict[str, Any]] = []
    for sr, ts, tmeas, substrate, psyn, pc, a_value in itertools.product(
        config.sr_values,
        config.ts_values,
        config.tmeas_values,
        config.substrates,
        config.psyn_mbar_values,
        config.pc_mbar_values,
        config.annealing_values,
    ):
        annealed = int(float(a_value)) == 1
        normalized_substrate = str(substrate).strip()
        rows.append(
            {
                "Psyn": float(psyn),
                "Oxygen activation": str(config.oxygen_activation).strip(),
                "Mismatch": mismatch_from_sr_substrate(float(sr), normalized_substrate),
                "Ts": float(ts),
                "A": int(float(a_value)),
                "TA": float(config.ta_deg_c) if annealed else None,
                "PA": float(config.pa_mbar) if annealed else None,
                "tA": float(config.ta_h) if annealed else None,
                "Pc": float(pc),
                "t": float(config.thickness_nm),
                "Sr": float(sr),
                "Tmeas": float(tmeas),
                "Growth method": str(config.growth_method).strip(),
                # Substrate is a search convenience used only to derive
                # Mismatch. It is intentionally not part of the model schema.
                "Substrate": normalized_substrate,
            }
        )
    if len(rows) != expected_count:  # pragma: no cover - defensive invariant
        raise RuntimeError("Candidate-grid size changed while building the grid")
    return pd.DataFrame(rows)


def rank_candidates(
    candidates: pd.DataFrame,
    predicted_rho_uohm_cm: Sequence[float],
    predicted_log10_rho_uohm_cm: Sequence[float],
    target_rho_uohm_cm: float,
) -> pd.DataFrame:
    """Attach prediction metrics and rank candidates deterministically."""

    target = _finite_number(target_rho_uohm_cm, "target_rho_uohm_cm")
    if target <= 0.0:
        raise ValueError("target_rho_uohm_cm must be positive")
    predicted = _finite_sequence(predicted_rho_uohm_cm, "predicted_rho_uohm_cm")
    predicted_log = _finite_sequence(
        predicted_log10_rho_uohm_cm, "predicted_log10_rho_uohm_cm"
    )
    if len(candidates) == 0:
        raise ValueError("candidates must not be empty")
    if len(predicted) != len(candidates) or len(predicted_log) != len(candidates):
        raise ValueError("Prediction lengths must match the candidate row count")
    if any(value <= 0.0 for value in predicted):
        raise ValueError("predicted_rho_uohm_cm values must be positive")

    target_log10 = math.log10(target)
    ranked = candidates.copy()
    ranked.insert(0, "Predicted_rho_uohm_cm", predicted)
    ranked.insert(1, "Target_rho_uohm_cm", target)
    ranked.insert(2, "Predicted_log10_rho_uohm_cm", predicted_log)
    ranked.insert(3, "Target_log10_rho_uohm_cm", target_log10)
    ranked.insert(
        4,
        "Abs_error_log10",
        np.abs(np.asarray(predicted_log, dtype=float) - target_log10),
    )
    ranked.insert(
        5,
        "Ratio_pred_over_target",
        np.asarray(predicted, dtype=float) / target,
    )
    ranked.insert(
        6,
        "Abs_relative_error_percent",
        np.abs(np.asarray(predicted, dtype=float) / target - 1.0) * 100.0,
    )
    ranked = ranked.sort_values(
        ["Abs_error_log10", "Abs_relative_error_percent"], kind="mergesort"
    ).reset_index(drop=True)
    ranked.insert(0, "Rank", np.arange(1, len(ranked) + 1, dtype=int))
    return ranked


def mismatch_table_records() -> list[dict[str, float]]:
    """Return the package's mismatch table as JSON/XLSX-friendly rows."""

    return [
        {
            "Sr": float(sr),
            **{
                substrate: float(MISMATCH_VALUES[substrate][index])
                for substrate in ("LAO", "STO", "LSAT", "MgO", "YSZ")
            },
        }
        for index, sr in enumerate(MISMATCH_SR)
    ]


def normalized_config(config: InverseDesignConfig) -> dict[str, Any]:
    """Serialize a validated configuration without filesystem paths."""

    candidate_count = validate_inverse_config(config)
    return {
        "target_rho_uohm_cm": float(config.target_rho_uohm_cm),
        "target_unit": "μΩ·cm",
        "top_n": int(config.top_n),
        "sr_values": [float(value) for value in config.sr_values],
        "ts_values": [float(value) for value in config.ts_values],
        "tmeas_values": [float(value) for value in config.tmeas_values],
        "substrates": [str(value).strip() for value in config.substrates],
        "psyn_mbar_values": [float(value) for value in config.psyn_mbar_values],
        "pc_mbar_values": [float(value) for value in config.pc_mbar_values],
        "growth_method": str(config.growth_method).strip(),
        "oxygen_activation": str(config.oxygen_activation).strip(),
        "annealing_values": [int(float(value)) for value in config.annealing_values],
        "ta_deg_c": float(config.ta_deg_c),
        "pa_mbar": float(config.pa_mbar),
        "ta_h": float(config.ta_h),
        "thickness_nm": float(config.thickness_nm),
        "candidate_count": candidate_count,
        "candidate_limit": MAX_INVERSE_CANDIDATES,
    }


def dataframe_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert a frame to strict-JSON records, replacing NaN with ``None``."""

    clean = frame.astype(object).where(pd.notna(frame), None)
    return clean.to_dict(orient="records")


__all__ = [
    "AIR_O2_MBAR",
    "INVERSE_DESIGN_NOTE",
    "InverseDesignConfig",
    "MAX_INVERSE_CANDIDATES",
    "TORR_TO_MBAR",
    "VALID_GROWTH_METHODS",
    "VALID_OXYGEN_ACTIVATIONS",
    "VALID_SUBSTRATES",
    "build_candidate_grid",
    "dataframe_records",
    "mismatch_from_sr_substrate",
    "mismatch_table_records",
    "normalized_config",
    "rank_candidates",
    "validate_inverse_config",
]
