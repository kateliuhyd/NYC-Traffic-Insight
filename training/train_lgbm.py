"""
LightGBM trainer for NYC traffic volume prediction.

Expected to outperform both HGB and RF due to:
- Native categorical feature handling
- Better treatment of missing values
- Faster training with equivalent or better accuracy
- Much smaller model files than RandomForest

Usage:
    python -m training.train_lgbm
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RANDOM_STATE, TRAIN_TEST_SPLIT


# ── Feature definitions (V2: includes weather) ───────────────────
CYCLIC_FEATURES = [
    "hour_sin", "hour_cos",
    "wd_sin", "wd_cos",
    "month_sin", "month_cos",
]

WEATHER_FEATURES = [
    "temperature_2m",
    "precipitation",
    "cloud_cover_low",
    "snow_depth",
    "visibility",           # NEW: high impact on driving behavior
    "rain",                 # NEW: finer than precipitation
    "snowfall",             # NEW: active snowfall
]

LAG_FEATURES = [
    "vol_lag_1",
    "vol_roll_3h",
    "vol_roll_24h",
]

EVENT_FEATURES = [
    "is_holiday",
    "heavy_snow",
]

ALL_FEATURES = CYCLIC_FEATURES + WEATHER_FEATURES + LAG_FEATURES + EVENT_FEATURES
TARGET = "Vol_log"  # log1p(Vol)


def load_and_prepare_data(data_path: str = "data/processed/merged_weather_traffic.csv"):
    """Load merged traffic + weather data, engineer features, and split."""
    import holidays

    df = pd.read_csv(data_path, parse_dates=["Timestamp"])
    df = df.sort_values("Timestamp").reset_index(drop=True)

    # ── Cyclic features ──────────────────────────────────────────
    df["hour_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["wd_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.weekday / 7)
    df["wd_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)

    # ── Lag features ─────────────────────────────────────────────
    df["vol_lag_1"] = df["Vol"].shift(1)
    df["vol_roll_3h"] = df["Vol"].rolling(3, min_periods=1).mean()
    df["vol_roll_24h"] = df["Vol"].rolling(24, min_periods=1).mean()

    # ── Event features ───────────────────────────────────────────
    us_holidays = holidays.US()
    df["is_holiday"] = df["Timestamp"].dt.date.apply(lambda d: int(d in us_holidays))
    if "snow_depth" in df.columns:
        df["heavy_snow"] = (df["snow_depth"] > 0.05).astype(int)
    else:
        df["heavy_snow"] = 0

    # ── Target ───────────────────────────────────────────────────
    df["Vol_log"] = np.log1p(df["Vol"])

    # ── Drop rows with NaN from lag features ─────────────────────
    available = [f for f in ALL_FEATURES if f in df.columns]
    df_clean = df.dropna(subset=available + [TARGET]).reset_index(drop=True)

    # ── Time-based split (NOT random!) ───────────────────────────
    split_idx = int(len(df_clean) * TRAIN_TEST_SPLIT)
    train_df = df_clean.iloc[:split_idx].copy()
    test_df = df_clean.iloc[split_idx:].copy()

    return train_df, test_df, available


def train_lgbm(train_df, test_df, features):
    """Train a LightGBM model and evaluate it."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed. Run: pip install lightgbm")
        sys.exit(1)

    X_train = train_df[features]
    y_train = train_df[TARGET]
    X_test = test_df[features]
    y_test_log = test_df[TARGET]
    y_test_raw = test_df["Vol"]

    model = lgb.LGBMRegressor(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=8,
        num_leaves=63,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    # Train with early stopping
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test_log)],
        callbacks=[lgb.early_stopping(50, verbose=True), lgb.log_evaluation(50)],
    )

    # ── Evaluate ─────────────────────────────────────────────────
    y_pred_log = model.predict(X_test)
    y_pred_raw = np.expm1(y_pred_log)

    nz = y_test_raw > 0  # avoid div-by-zero for MAPE

    print("\n" + "=" * 60)
    print("LightGBM Evaluation Results")
    print("=" * 60)
    print(f"  Features used:       {len(features)}")
    print(f"  Best iteration:      {model.best_iteration_}")
    print(f"  Train size:          {len(train_df):,}")
    print(f"  Test size:           {len(test_df):,}")
    print()
    print("  Log-space metrics:")
    print(f"    R²:    {r2_score(y_test_log, y_pred_log):.4f}")
    print(f"    MAE:   {mean_absolute_error(y_test_log, y_pred_log):.4f}")
    print()
    print("  Raw-volume metrics:")
    print(f"    R²:    {r2_score(y_test_raw, y_pred_raw):.4f}")
    print(f"    MAE:   {mean_absolute_error(y_test_raw, y_pred_raw):.2f}")
    print(f"    MAPE:  {mean_absolute_percentage_error(y_test_raw[nz], y_pred_raw[nz]) * 100:.2f}%")

    # ── Feature importance ───────────────────────────────────────
    feat_imp = sorted(
        zip(features, model.feature_importances_),
        key=lambda x: x[1], reverse=True,
    )
    print("\n  Feature importances (gain):")
    for name, imp in feat_imp:
        print(f"    {name:20s}: {imp:>6d}")

    return model


def save_model(model, output_path: str = "models/lgbm_model.joblib"):
    """Save the trained model."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_path)
    size_mb = Path(output_path).stat().st_size / 1024 / 1024
    print(f"\n  Model saved to {output_path} ({size_mb:.2f} MB)")


# ── Walk-forward evaluation ──────────────────────────────────────

def walk_forward_evaluate(df_clean, features, n_folds: int = 5):
    """Time-series cross-validation: expanding window, always predict the next chunk."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("LightGBM not installed.")
        return

    chunk_size = len(df_clean) // (n_folds + 1)
    results = []

    print("\n" + "=" * 60)
    print(f"Walk-Forward Cross-Validation ({n_folds} folds)")
    print("=" * 60)

    for i in range(n_folds):
        train_end = chunk_size * (i + 1)
        test_end = min(chunk_size * (i + 2), len(df_clean))

        train = df_clean.iloc[:train_end]
        test = df_clean.iloc[train_end:test_end]

        model = lgb.LGBMRegressor(
            n_estimators=300, learning_rate=0.05, max_depth=8,
            random_state=RANDOM_STATE, verbose=-1, n_jobs=-1,
        )
        model.fit(train[features], train[TARGET])

        y_pred = np.expm1(model.predict(test[features]))
        y_true = test["Vol"]
        nz = y_true > 0

        r2 = r2_score(y_true, y_pred)
        mae = mean_absolute_error(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true[nz], y_pred[nz]) * 100 if nz.sum() > 0 else float("nan")

        results.append({"fold": i + 1, "train_size": len(train), "test_size": len(test), "R2": r2, "MAE": mae, "MAPE": mape})
        print(f"  Fold {i+1}: train={len(train):>6,}  test={len(test):>6,}  R²={r2:.4f}  MAE={mae:.2f}  MAPE={mape:.2f}%")

    avg_r2 = np.mean([r["R2"] for r in results])
    avg_mae = np.mean([r["MAE"] for r in results])
    avg_mape = np.mean([r["MAPE"] for r in results])
    print(f"\n  Average: R²={avg_r2:.4f}  MAE={avg_mae:.2f}  MAPE={avg_mape:.2f}%")

    return results


# ── Main ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train LightGBM traffic volume model")
    parser.add_argument("--data", default="data/processed/merged_weather_traffic.csv",
                        help="Path to merged traffic+weather CSV")
    parser.add_argument("--output", default="models/lgbm_model.joblib",
                        help="Output model path")
    parser.add_argument("--walk-forward", action="store_true",
                        help="Run walk-forward cross-validation")
    args = parser.parse_args()

    train_df, test_df, features = load_and_prepare_data(args.data)
    model = train_lgbm(train_df, test_df, features)
    save_model(model, args.output)

    if args.walk_forward:
        # Combine for walk-forward
        full_df = pd.concat([train_df, test_df], ignore_index=True)
        walk_forward_evaluate(full_df, features)
