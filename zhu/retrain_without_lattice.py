"""Reproducibly train an LSCO model without the lattice parameters ``a``/``c``.

The production API expects a joblib dictionary containing ``model``,
``scaler``, ``imputer`` and ``selected_features``.  This script deliberately
creates that same contract, but never overwrites an existing artifact.  It is
an all-data retrain, not an XGBoost continuation/fine-tune.

Example
-------
python3 zhu/retrain_without_lattice.py data.xlsx models/lsco_without_lattice.joblib --target-unit ohm_cm

The source target is converted to the API's canonical ``μΩ·cm`` before
``log10`` is applied.  Therefore the API can continue to expose
``y = 10 ** y_log`` in ``μΩ·cm``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import MinMaxScaler


# These are the only features in the retrained model.  ``a`` and ``c`` are
# intentionally absent even when the source sheet contains those columns.
FEATURES: tuple[str, ...] = (
    "Psyn",
    "Ts",
    "A",
    "TA",
    "PA",
    "tA",
    "Pc",
    "t",
    "Sr",
    "Tmeas",
)
EXCLUDED_LATTICE_FEATURES: tuple[str, ...] = ("a", "c")
TARGET_UNIT_API = "μΩ·cm"
TARGET_LOG_UNIT_API = "log10(μΩ·cm)"
MIN_ROWS = 30
MIN_CONDITIONS = 5
RANDOM_SEED = 42


@dataclass(frozen=True)
class TrainingConfig:
    """Explicit model settings persisted in the resulting artifact."""

    n_estimators: int = 499
    learning_rate: float = 0.20808419644031184
    max_depth: int = 5
    subsample: float = 0.8415244766212273
    colsample_bytree: float = 0.8415244766212273
    n_jobs: int = 1
    random_state: int = RANDOM_SEED

    def xgboost_params(self) -> dict[str, Any]:
        return {
            "objective": "reg:squarederror",
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
            "tree_method": "hist",
        }


@dataclass(frozen=True)
class PreparedTrainingData:
    """Validated model inputs and canonical target values."""

    features: pd.DataFrame
    target_uohm_cm: pd.Series
    groups: pd.Series
    source_row_count: int
    source_target_unit: str
    source_columns: tuple[str, ...]
    constant_features: tuple[str, ...]


def _read_dataframe(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
    elif suffix in {".xlsx", ".xls"}:
        frame = pd.read_excel(path)
    else:
        raise ValueError("训练数据仅支持 CSV、XLS 或 XLSX 文件")

    if frame.empty:
        raise ValueError("训练数据为空")
    normalized_columns = [str(column).strip() for column in frame.columns]
    if len(set(normalized_columns)) != len(normalized_columns):
        raise ValueError("训练数据包含重复列名")
    return frame.rename(columns=dict(zip(frame.columns, normalized_columns)))


def _finite_numeric_columns(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    """Convert required columns and reject every missing/non-numeric value."""

    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError("训练数据缺少必需特征列：" + "、".join(missing))

    numeric = pd.DataFrame(index=frame.index)
    errors: list[str] = []
    for column in columns:
        raw = frame[column]
        converted = pd.to_numeric(raw, errors="coerce")
        empty_count = int(raw.isna().sum())
        non_numeric_count = int((raw.notna() & converted.isna()).sum())
        non_finite_count = int(
            (converted.notna() & ~np.isfinite(converted.astype(float))).sum()
        )
        if empty_count or non_numeric_count or non_finite_count:
            errors.append(
                f"{column}(空值={empty_count}, 非数值={non_numeric_count}, 非有限值={non_finite_count})"
            )
        numeric[column] = converted.astype(float)

    if errors:
        raise ValueError("训练特征必须全部为有限数值：" + "；".join(errors))
    return numeric


def _prepare_target(
    frame: pd.DataFrame,
    target_column: str,
    source_target_unit: Literal["ohm_cm", "uohm_cm"],
) -> pd.Series:
    if target_column not in frame.columns:
        raise ValueError(f"训练数据缺少目标列：{target_column}")

    raw = frame[target_column]
    target = pd.to_numeric(raw, errors="coerce")
    empty_count = int(raw.isna().sum())
    non_numeric_count = int((raw.notna() & target.isna()).sum())
    non_finite_count = int(
        (target.notna() & ~np.isfinite(target.astype(float))).sum()
    )
    if empty_count or non_numeric_count or non_finite_count:
        raise ValueError(
            "目标列必须全部为有限数值："
            f"空值={empty_count}, 非数值={non_numeric_count}, 非有限值={non_finite_count}"
        )

    factor = 1_000_000.0 if source_target_unit == "ohm_cm" else 1.0
    target_uohm_cm = target.astype(float) * factor
    if not bool(np.isfinite(target_uohm_cm).all()):
        raise ValueError(f"目标列换算为 {TARGET_UNIT_API} 后包含非有限值")
    if not bool((target_uohm_cm > 0.0).all()):
        invalid_count = int((target_uohm_cm <= 0.0).sum())
        raise ValueError(
            f"目标电阻率必须为正数；发现 {invalid_count} 个非正值（换算为 {TARGET_UNIT_API} 后）"
        )
    return target_uohm_cm.rename("rho_uohm_cm")


def _condition_groups(feature_frame: pd.DataFrame) -> pd.Series:
    """Group rows from the same sample condition so temperature curves do not leak.

    A curve has several ``Tmeas`` rows but one shared fabrication condition.
    Grouped cross-validation keeps all rows of a condition in the same fold.
    Because ``a`` and ``c`` are excluded from the model, they also cannot
    create artificial split groups here.
    """

    condition_columns = [column for column in FEATURES if column != "Tmeas"]
    hashes = pd.util.hash_pandas_object(
        feature_frame.loc[:, condition_columns], index=False
    )
    return hashes.astype("uint64").astype(str).rename("condition_group")


def prepare_training_data(
    data_path: str | os.PathLike[str],
    *,
    target_column: str = "ρ",
    source_target_unit: Literal["ohm_cm", "uohm_cm"],
) -> PreparedTrainingData:
    """Read, validate and canonicalize a dataset for the ten-feature model."""

    if source_target_unit not in {"ohm_cm", "uohm_cm"}:
        raise ValueError("source_target_unit 只能是 ohm_cm 或 uohm_cm")

    path = Path(data_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"训练数据不存在：{path}")

    frame = _read_dataframe(path)
    if len(frame) < MIN_ROWS:
        raise ValueError(
            f"训练数据至少需要 {MIN_ROWS} 行；当前仅 {len(frame)} 行。"
            "这可防止用单条或极少数温度曲线生成不可靠模型。"
        )

    feature_frame = _finite_numeric_columns(frame, FEATURES)
    # A is not merely a number: it has the known 0/1 physical encoding.
    invalid_a_count = int((~feature_frame["A"].isin([0.0, 1.0])).sum())
    if invalid_a_count:
        raise ValueError(f"A 必须是退火状态 0 或 1；发现 {invalid_a_count} 个其他值")

    target_uohm_cm = _prepare_target(frame, target_column, source_target_unit)
    groups = _condition_groups(feature_frame)
    group_count = int(groups.nunique())
    if group_count < MIN_CONDITIONS:
        raise ValueError(
            f"训练数据至少需要 {MIN_CONDITIONS} 个不同的制备条件（不含 Tmeas）；"
            f"当前仅 {group_count} 个。"
        )

    constant_features = tuple(
        column for column in FEATURES if int(feature_frame[column].nunique()) <= 1
    )
    return PreparedTrainingData(
        features=feature_frame,
        target_uohm_cm=target_uohm_cm,
        groups=groups,
        source_row_count=int(len(frame)),
        source_target_unit=source_target_unit,
        source_columns=tuple(str(column) for column in frame.columns),
        constant_features=constant_features,
    )


def _fit_components(
    features: pd.DataFrame,
    y_log: pd.Series | np.ndarray,
    config: TrainingConfig,
) -> tuple[SimpleImputer, MinMaxScaler, xgb.XGBRegressor]:
    """Fit preprocessing and model in the exact order used by the API."""

    imputer = SimpleImputer(strategy="median")
    imputed = imputer.fit_transform(features.loc[:, FEATURES])
    imputed_frame = pd.DataFrame(imputed, columns=FEATURES, index=features.index)
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(imputed_frame)
    model = xgb.XGBRegressor(**config.xgboost_params())
    model.fit(scaled, np.asarray(y_log, dtype=float))
    return imputer, scaler, model


def _predict_components(
    features: pd.DataFrame,
    imputer: SimpleImputer,
    scaler: MinMaxScaler,
    model: xgb.XGBRegressor,
) -> np.ndarray:
    imputed = imputer.transform(features.loc[:, FEATURES])
    imputed_frame = pd.DataFrame(imputed, columns=FEATURES, index=features.index)
    return np.asarray(model.predict(scaler.transform(imputed_frame)), dtype=float)


def _cross_validated_metrics(
    prepared: PreparedTrainingData, config: TrainingConfig
) -> tuple[dict[str, float | int], np.ndarray]:
    """Produce group-held-out out-of-fold predictions and metrics in two units."""

    y_log = np.log10(prepared.target_uohm_cm.to_numpy(dtype=float))
    groups = prepared.groups.to_numpy()
    group_count = int(prepared.groups.nunique())
    fold_count = min(5, group_count)
    splitter = GroupKFold(n_splits=fold_count)
    oof_log = np.full(len(prepared.features), np.nan, dtype=float)

    for train_index, valid_index in splitter.split(prepared.features, y_log, groups):
        imputer, scaler, model = _fit_components(
            prepared.features.iloc[train_index], y_log[train_index], config
        )
        oof_log[valid_index] = _predict_components(
            prepared.features.iloc[valid_index], imputer, scaler, model
        )

    if not bool(np.isfinite(oof_log).all()):
        raise RuntimeError("交叉验证未为每一行生成有限预测值")

    y_linear = prepared.target_uohm_cm.to_numpy(dtype=float)
    oof_linear = np.power(10.0, oof_log)
    metrics: dict[str, float | int] = {
        "cv_strategy": "GroupKFold by fabrication condition excluding Tmeas",
        "cv_folds": fold_count,
        "cv_r2_log": float(r2_score(y_log, oof_log)),
        "cv_rmse_log": float(math.sqrt(mean_squared_error(y_log, oof_log))),
        "cv_mae_log": float(mean_absolute_error(y_log, oof_log)),
        "cv_rmse_uohm_cm": float(math.sqrt(mean_squared_error(y_linear, oof_linear))),
        "cv_mae_uohm_cm": float(mean_absolute_error(y_linear, oof_linear)),
    }
    return metrics, oof_log


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retrain_without_lattice(
    data_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    *,
    target_column: str = "ρ",
    source_target_unit: Literal["ohm_cm", "uohm_cm"],
    config: TrainingConfig | None = None,
) -> Mapping[str, Any]:
    """Train a new ten-feature joblib bundle without touching existing models.

    The output filename must be new.  It is intentionally not permitted to
    replace the application default model, so changing production model paths
    remains a separate explicit deployment decision.
    """

    config = config or TrainingConfig()
    source_path = Path(data_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    default_model_path = (
        Path(__file__).resolve().parent
        / "models"
        / "NGO_XGBoost_package_full_pipeline.joblib"
    ).resolve()
    if destination.suffix.lower() != ".joblib":
        raise ValueError("输出文件必须以 .joblib 结尾")
    if destination == default_model_path:
        raise ValueError("为保护现有线上模型，输出路径不能是默认模型文件")
    if destination.exists():
        raise FileExistsError(f"输出文件已存在，拒绝覆盖：{destination}")

    prepared = prepare_training_data(
        source_path,
        target_column=target_column,
        source_target_unit=source_target_unit,
    )
    metrics, oof_log = _cross_validated_metrics(prepared, config)
    y_log = np.log10(prepared.target_uohm_cm.to_numpy(dtype=float))
    imputer, scaler, model = _fit_components(prepared.features, y_log, config)

    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "schema_version": 2,
        "model_label": destination.name,
        "created_utc": datetime.now(UTC).isoformat(),
        "source_file": source_path.name,
        "source_sha256": _sha256_file(source_path),
        "source_row_count": prepared.source_row_count,
        "source_columns": list(prepared.source_columns),
        "target_column": target_column,
        "source_target_unit": source_target_unit,
        "api_target_unit": TARGET_UNIT_API,
        "api_target_log_unit": TARGET_LOG_UNIT_API,
        "target_conversion": (
            "rho_uohm_cm = rho_source * 1e6"
            if source_target_unit == "ohm_cm"
            else "rho_uohm_cm = rho_source"
        ),
        "target_transform": "y_log = log10(rho_uohm_cm)",
        "selected_features": list(FEATURES),
        "excluded_features": list(EXCLUDED_LATTICE_FEATURES),
        "condition_grouping": "All selected features except Tmeas",
        "unique_conditions": int(prepared.groups.nunique()),
        "constant_features": list(prepared.constant_features),
        "training_config": asdict(config),
        "library_versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": sklearn.__version__,
            "xgboost": xgb.__version__,
        },
    }
    bundle: dict[str, Any] = {
        # These four keys are required by zhu/backend_app.py.
        "model": model,
        "scaler": scaler,
        "imputer": imputer,
        "selected_features": list(FEATURES),
        # Preserve a familiar key for existing tooling and record all details
        # needed to audit/reproduce the retrain.
        "best_params": config.xgboost_params(),
        "metadata": metadata,
        "metrics": metrics,
        "oof_predictions": {
            "y_true_log": y_log.tolist(),
            "y_pred_log": oof_log.tolist(),
        },
    }
    joblib.dump(bundle, destination)
    return {
        "output_path": str(destination),
        "selected_features": list(FEATURES),
        "excluded_features": list(EXCLUDED_LATTICE_FEATURES),
        "metrics": metrics,
        "metadata": metadata,
    }


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Retrain an LSCO resistivity model without a/c lattice parameters."
    )
    parser.add_argument("data_path", help="CSV/XLS/XLSX training table")
    parser.add_argument("output_path", help="new .joblib output path; existing files are never overwritten")
    parser.add_argument("--target-column", default="ρ", help="positive resistivity target column (default: ρ)")
    parser.add_argument(
        "--target-unit",
        choices=("ohm_cm", "uohm_cm"),
        required=True,
        help="unit stored in the source target column; this must be declared explicitly",
    )
    parser.add_argument("--n-estimators", type=int, default=TrainingConfig.n_estimators)
    parser.add_argument("--learning-rate", type=float, default=TrainingConfig.learning_rate)
    parser.add_argument("--max-depth", type=int, default=TrainingConfig.max_depth)
    parser.add_argument("--subsample", type=float, default=TrainingConfig.subsample)
    parser.add_argument("--colsample-bytree", type=float, default=TrainingConfig.colsample_bytree)
    parser.add_argument("--n-jobs", type=int, default=TrainingConfig.n_jobs)
    return parser


def main() -> int:
    parser = _build_argument_parser()
    args = parser.parse_args()
    config = TrainingConfig(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        n_jobs=args.n_jobs,
    )
    try:
        result = retrain_without_lattice(
            args.data_path,
            args.output_path,
            target_column=args.target_column,
            source_target_unit=args.target_unit,
            config=config,
        )
    except (FileNotFoundError, FileExistsError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))

    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
