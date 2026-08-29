# -*- coding: utf-8 -*-
import os
from collections.abc import Mapping
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def _read_training_data(path: str) -> pd.DataFrame:
    """Read the CSV or Excel training data accepted by the fine-tuning UI."""
    suffix = os.path.splitext(os.fspath(path))[1].lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError("New data must be a .csv, .xlsx, or .xls file")


def _encode_model_features(
    frame: pd.DataFrame,
    selected_features: list[str],
    category_mappings: Mapping[str, Mapping[str, object]],
    conditional_missing_when_a_zero: tuple[str, ...] = (),
    input_transforms: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Encode browser fields into the numeric matrix used by the model.

    The supplied Feuil5 data records the annealing-only fields as missing when
    ``A=0``.  The browser keeps those cells visibly complete by using ``0``;
    mirror the prediction API here so fine-tuning sees the exact same saved
    imputer behavior as deployment.
    """

    missing = [feature for feature in selected_features if feature not in frame.columns]
    if missing:
        raise ValueError("New data is missing model fields: " + ", ".join(missing))

    encoded = pd.DataFrame(index=frame.index)
    for feature in selected_features:
        raw = frame[feature]
        mapping = category_mappings.get(feature)
        if mapping:
            allowed_codes = {float(value) for value in mapping.values()}

            def encode(value):
                if pd.isna(value):
                    return np.nan
                label = str(value).strip()
                if label in mapping:
                    return float(mapping[label])
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return np.nan
                return number if np.isfinite(number) and number in allowed_codes else np.nan

            encoded[feature] = raw.map(encode).astype(float)
        else:
            encoded[feature] = pd.to_numeric(raw, errors="coerce").astype(float)

    if "A" in encoded.columns:
        unannealed = encoded["A"].eq(0.0)
        for feature in conditional_missing_when_a_zero:
            if feature in encoded.columns and feature != "A":
                encoded.loc[unannealed, feature] = np.nan

    for feature, transform in (input_transforms or {}).items():
        if feature not in encoded.columns:
            raise ValueError(f"Model input transform refers to an unknown field: {feature}")
        if transform != "log10_positive":
            raise ValueError(f"Unsupported input transform for {feature}: {transform}")
        invalid = encoded[feature].notna() & (encoded[feature] <= 0.0)
        if invalid.any():
            raise ValueError(f"{feature} must contain positive values before log10 preprocessing")
        encoded[feature] = np.log10(encoded[feature].where(encoded[feature] > 0.0))
    return encoded


def fine_tune_xgboost_pipeline(
    old_model_path: str,
    new_data_path: str,
    save_path: str,
    num_round: int = 50,
    learning_rate: float = 0.01,
    verbose: bool = True
):
    """
    Fine-tune a saved XGBoost pipeline model (.joblib) with new data.

    Parameters
    ----------
    old_model_path : str
        Path to the existing saved model (.joblib)
    new_data_path : str
        Path to new CSV or Excel file containing training data (last column =
        resistivity target in μΩ·cm)
    save_path : str
        Path to save the fine-tuned model
    num_round : int, optional
        Number of boosting rounds to add during fine-tuning (default=50)
    learning_rate : float, optional
        Learning rate for fine-tuning (default=0.01)
    verbose : bool, optional
        Print progress logs (default=True)

    Returns
    -------
    dict
        Dictionary containing before/after metrics and model info.
    """

    # ---------- Helper ----------
    def log(msg):
        if verbose:
            print(msg)

    def convert_to_float(x):
        """Convert ranges and common resistivity-unit suffixes to floats."""
        if isinstance(x, str):
            x = (
                x.strip()
                .replace(",", "")
                .replace("μΩ·cm", "")
                .replace("uΩ·cm", "")
                .replace("Ω·cm", "")
                .replace("ohm·cm", "")
            )
            if "-" in x:
                try:
                    low, high = map(float, x.split("-"))
                    return (low + high) / 2
                except:
                    pass
            try:
                return float(x)
            except:
                return np.nan
        return x

    # ---------- 1. Load old model ----------
    bundle = joblib.load(old_model_path)
    xgb_model = bundle["model"]
    scaler = bundle["scaler"]
    imputer = bundle["imputer"]
    selected_features = list(bundle.get("selected_features", bundle.get("features", [])))
    if not selected_features:
        raise ValueError("The model package is missing selected_features or features")
    model_feature_columns = list(bundle.get("model_feature_columns", selected_features))
    if len(model_feature_columns) != len(selected_features):
        raise ValueError("model_feature_columns and selected_features have different lengths")
    raw_categories = bundle.get("category_mappings", {})
    if not isinstance(raw_categories, Mapping):
        raise ValueError("Model category_mappings has an invalid format")
    category_mappings = {
        str(feature): {str(label): value for label, value in mapping.items()}
        for feature, mapping in raw_categories.items()
        if isinstance(mapping, Mapping)
    }
    raw_metadata = bundle.get("metadata", {})
    metadata = raw_metadata if isinstance(raw_metadata, Mapping) else {}
    output_log_offset = float(metadata.get("output_log_offset", bundle.get("output_log_offset", 0.0)))
    raw_conditional = metadata.get("conditional_missing_when_a_zero", ())
    if not isinstance(raw_conditional, (list, tuple)) or not all(
        isinstance(feature, str) for feature in raw_conditional
    ):
        raise ValueError("Model conditional_missing_when_a_zero has an invalid format")
    conditional_missing_when_a_zero = tuple(
        feature for feature in raw_conditional if feature in selected_features and feature != "A"
    )
    raw_transforms = metadata.get("input_transforms", bundle.get("input_transforms", {}))
    if not isinstance(raw_transforms, Mapping):
        raise ValueError("Model input_transforms has an invalid format")
    input_transforms = {
        str(feature): str(transform) for feature, transform in raw_transforms.items()
    }

    log("✅ 已加载旧模型和预处理器")

    # ---------- 2. Load new data ----------
    df_new = _read_training_data(new_data_path)
    log(f"📄 新数据维度: {df_new.shape}")
    log(df_new.head(2))

    # Clean target column
    y_new_raw = df_new.iloc[:, -1].map(convert_to_float)
    y_new = y_new_raw.astype(float).replace([np.inf, -np.inf], np.nan).dropna()
    X_new = df_new.iloc[:, :-1].copy().loc[y_new.index]
    # The saved model predicts log10(ρ / (μΩ·cm)).  Preserve that unit when
    # fine-tuning so the API can consistently return y = 10**y_log in μΩ·cm.
    y_new_log_api = np.log10(y_new + 1e-12)
    y_new_log = y_new_log_api - output_log_offset

    # ---------- 3. Preprocessing ----------
    X_new_encoded = _encode_model_features(
        X_new,
        selected_features,
        category_mappings,
        conditional_missing_when_a_zero,
        input_transforms,
    )
    X_new_encoded.columns = model_feature_columns
    X_new_imp = imputer.transform(X_new_encoded)
    X_new_imp_frame = pd.DataFrame(X_new_imp, columns=model_feature_columns, index=X_new.index)
    scaler_input = X_new_imp_frame if hasattr(scaler, "feature_names_in_") else X_new_imp
    X_new_scaled = scaler.transform(scaler_input)
    X_new_sel = np.asarray(X_new_scaled, dtype=float)
    dtrain_new = xgb.DMatrix(X_new_sel, label=y_new_log)
    log("✅ 新数据已完成预处理")

    # ---------- 4. Evaluate old model ----------
    y_pred_old = xgb_model.predict(X_new_sel)
    r2_old = r2_score(y_new_log, y_pred_old)
    rmse_old = float(np.sqrt(mean_squared_error(y_new_log, y_pred_old)))
    mae_old = mean_absolute_error(y_new_log, y_pred_old)
    log(f"\n📊 [旧模型在新数据上的表现]\nR² = {r2_old:.4f}, RMSE = {rmse_old:.4f}, MAE = {mae_old:.4f}")

    # ---------- 5. Fine-tuning ----------
    booster = xgb_model.get_booster()

    params = xgb_model.get_xgb_params()
    params["learning_rate"] = learning_rate

    updated_booster = xgb.train(
        params=params,
        dtrain=dtrain_new,
        num_boost_round=num_round,
        # Passing the in-memory Booster avoids a shared temp_old_model.json.
        xgb_model=booster,
    )

    num_trees_old = len(booster.get_dump())
    num_trees_new = len(updated_booster.get_dump())
    log(f"\n🌲 树数量变化: {num_trees_old} → {num_trees_new} (+{num_trees_new - num_trees_old})")

    # ---------- 6. Evaluate fine-tuned model ----------
    y_pred_new = updated_booster.predict(dtrain_new)
    r2_new = r2_score(y_new_log, y_pred_new)
    rmse_new = float(np.sqrt(mean_squared_error(y_new_log, y_pred_new)))
    mae_new = mean_absolute_error(y_new_log, y_pred_new)
    log(f"\n📊 [微调后模型在新数据上的表现]\nR² = {r2_new:.4f}, RMSE = {rmse_new:.4f}, MAE = {mae_new:.4f}")

    # ---------- 7. Save model ----------
    xgb_model._Booster = updated_booster
    bundle["model"] = xgb_model
    save_path = os.path.abspath(os.path.expanduser(save_path))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump(bundle, save_path)
    log(f"\n✅ 新模型已保存到:\n{save_path}")

    # ---------- 8. Export prediction differences ----------
    diff_df = pd.DataFrame({
        "y_true_log_uohm_cm": y_new_log_api,
        "y_pred_old_log_uohm_cm": y_pred_old + output_log_offset,
        "y_pred_new_log_uohm_cm": y_pred_new + output_log_offset,
        "diff_log": y_pred_new - y_pred_old,
    })
    diff_path = os.path.splitext(save_path)[0] + "_comparison.csv"
    diff_df.to_csv(diff_path, index=False)
    log(f"📄 微调前后预测差异已导出到:\n{diff_path}")

    # ---------- 9. Return summary ----------
    summary = {
        "R2_before": r2_old,
        "R2_after": r2_new,
        "RMSE_before": rmse_old,
        "RMSE_after": rmse_new,
        "MAE_before": mae_old,
        "MAE_after": mae_new,
        "num_trees_before": num_trees_old,
        "num_trees_after": num_trees_new,
        "delta_R2": r2_new - r2_old,
        "delta_RMSE": rmse_new - rmse_old,
        "delta_MAE": mae_new - mae_old,
        "comparison_csv": diff_path,
        "fine_tuned_model_path": save_path
    }

    return summary


# ========== Command-line entry point ==========
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Fine-tune the LSCO XGBoost model.")
    parser.add_argument("model_path", help="Existing .joblib model package")
    parser.add_argument("data_path", help="CSV, XLS, or XLSX data; last column is rho in μΩ·cm")
    parser.add_argument("save_path", help="Output path for the fine-tuned .joblib model")
    parser.add_argument("--rounds", type=int, default=50, help="Additional boosting rounds (default: 50)")
    parser.add_argument("--learning-rate", type=float, default=0.01, help="Fine-tuning learning rate (default: 0.01)")
    args = parser.parse_args()

    summary = fine_tune_xgboost_pipeline(
        old_model_path=args.model_path,
        new_data_path=args.data_path,
        save_path=args.save_path,
        num_round=args.rounds,
        learning_rate=args.learning_rate,
    )

    print("\n===== 微调总结 =====")
    for k, v in summary.items():
        print(f"{k:25s}: {v}")
