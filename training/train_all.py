"""
Unified training pipeline — train all models with consistent settings.

This script ensures all models use:
    - The same data source
    - Time-based train/test split (NOT random)
    - The same feature engineering pipeline
    - Consistent evaluation metrics

Usage:
    python -m training.train_all                        # train all models
    python -m training.train_all --models hgb lgbm      # train specific models
    python -m training.train_all --walk-forward          # include WF evaluation
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config import RANDOM_STATE, TRAIN_TEST_SPLIT


# ── Shared feature engineering ───────────────────────────────────

CYCLIC = ["hour_sin", "hour_cos", "wd_sin", "wd_cos", "month_sin", "month_cos"]
LAG = ["vol_lag_1", "vol_roll_3h", "vol_roll_24h"]
EVENT = ["is_holiday", "heavy_snow"]
WEATHER = ["temperature_2m", "precipitation", "cloud_cover_low", "snow_depth"]
TARGET = "Vol_log"


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature engineering to a raw merged DataFrame."""
    import holidays as holidays_lib

    df = df.sort_values("Timestamp").reset_index(drop=True)

    # Cyclic encoding
    df["hour_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.hour / 24)
    df["wd_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.weekday / 7)
    df["wd_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.weekday / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["Timestamp"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["Timestamp"].dt.month / 12)

    # Lag features
    df["vol_lag_1"] = df["Vol"].shift(1)
    df["vol_roll_3h"] = df["Vol"].rolling(3, min_periods=1).mean()
    df["vol_roll_24h"] = df["Vol"].rolling(24, min_periods=1).mean()

    # Event features
    us_hol = holidays_lib.US()
    df["is_holiday"] = df["Timestamp"].dt.date.apply(lambda d: int(d in us_hol))
    df["heavy_snow"] = (df.get("snow_depth", pd.Series(0, index=df.index)) > 0.05).astype(int)

    # Target
    df["Vol_log"] = np.log1p(df["Vol"])

    return df.dropna(subset=CYCLIC + LAG + [TARGET]).reset_index(drop=True)


def split_data(df: pd.DataFrame):
    """Time-based train/test split."""
    idx = int(len(df) * TRAIN_TEST_SPLIT)
    return df.iloc[:idx].copy(), df.iloc[idx:].copy()


def evaluate(y_true_raw, y_pred_raw, label: str = ""):
    """Print standard metrics."""
    nz = y_true_raw > 0
    r2 = r2_score(y_true_raw, y_pred_raw)
    mae = mean_absolute_error(y_true_raw, y_pred_raw)
    mape = mean_absolute_percentage_error(y_true_raw[nz], y_pred_raw[nz]) * 100

    prefix = f"  [{label}] " if label else "  "
    print(f"{prefix}R²: {r2:.4f}  |  MAE: {mae:.2f}  |  MAPE: {mape:.2f}%")
    return {"R2": r2, "MAE": mae, "MAPE": mape}


# ── Model trainers ───────────────────────────────────────────────

def train_hgb(train_df, test_df, features=None):
    """Train HistGradientBoostingRegressor."""
    from sklearn.ensemble import HistGradientBoostingRegressor

    feats = features or CYCLIC + LAG
    model = HistGradientBoostingRegressor(
        max_iter=200, learning_rate=0.1, max_depth=6,
        early_stopping=True, random_state=RANDOM_STATE,
    )
    model.fit(train_df[feats], train_df[TARGET])

    y_pred = np.expm1(model.predict(test_df[feats]))
    print("\nHistGradientBoosting:")
    evaluate(test_df["Vol"].values, y_pred, "HGB")
    return model, feats


def train_lgbm(train_df, test_df, features=None):
    """Train LightGBM."""
    try:
        import lightgbm as lgb
    except ImportError:
        print("⚠️  LightGBM not installed, skipping. Run: pip install lightgbm")
        return None, []

    feats = features or CYCLIC + WEATHER + LAG + EVENT
    feats = [f for f in feats if f in train_df.columns]

    model = lgb.LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=8,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        reg_alpha=0.1, reg_lambda=0.1,
        random_state=RANDOM_STATE, n_jobs=-1, verbose=-1,
    )
    model.fit(
        train_df[feats], train_df[TARGET],
        eval_set=[(test_df[feats], test_df[TARGET])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    y_pred = np.expm1(model.predict(test_df[feats]))
    print("\nLightGBM:")
    evaluate(test_df["Vol"].values, y_pred, "LGBM")
    return model, feats


def train_segmented(train_df, test_df, features=None):
    """Train SegmentedModel."""
    from src.models.segmented_model import SegmentedModel

    feats = features or CYCLIC + LAG
    model = SegmentedModel()
    # SegmentedModel handles its own split, so pass full data
    full = pd.concat([train_df, test_df], ignore_index=True)
    model.fit(full, feats, target_col=TARGET)
    model.evaluate()
    return model, feats


# ── Walk-forward evaluation ──────────────────────────────────────

def walk_forward(df, features, trainer_fn, n_folds=5, label=""):
    """Expanding-window walk-forward CV."""
    chunk = len(df) // (n_folds + 1)
    scores = []

    print(f"\n{'=' * 50}")
    print(f"Walk-Forward CV for {label} ({n_folds} folds)")
    print(f"{'=' * 50}")

    for i in range(n_folds):
        t_end = chunk * (i + 1)
        v_end = min(chunk * (i + 2), len(df))
        train, test = df.iloc[:t_end], df.iloc[t_end:v_end]

        model, feats = trainer_fn(train, test, features)
        if model is None:
            break

        if hasattr(model, "predict") and hasattr(model, "__class__"):
            if model.__class__.__name__ == "SegmentedModel":
                y_pred = model.predict(test)
            else:
                y_pred = np.expm1(model.predict(test[feats]))
        else:
            break

        s = evaluate(test["Vol"].values, y_pred, f"Fold {i+1}")
        scores.append(s)

    if scores:
        print(f"\n  Average: R²={np.mean([s['R2'] for s in scores]):.4f}  "
              f"MAE={np.mean([s['MAE'] for s in scores]):.2f}  "
              f"MAPE={np.mean([s['MAPE'] for s in scores]):.2f}%")


# ── Main entry ───────────────────────────────────────────────────

TRAINERS = {
    "hgb": ("HistGradientBoosting", train_hgb, "models/hgb_model.joblib"),
    "lgbm": ("LightGBM", train_lgbm, "models/lgbm_model.joblib"),
    "seg": ("Segmented HGB", train_segmented, "models/segmented_model.joblib"),
}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Unified NYC Traffic model trainer")
    parser.add_argument("--data", default="data/processed/merged_weather_traffic.csv")
    parser.add_argument("--models", nargs="*", default=list(TRAINERS.keys()),
                        help=f"Models to train: {list(TRAINERS.keys())}")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    print("Loading and preparing data...")
    df = pd.read_csv(args.data, parse_dates=["Timestamp"])
    df = engineer_features(df)
    train_df, test_df = split_data(df)
    print(f"  Total: {len(df):,}  Train: {len(train_df):,}  Test: {len(test_df):,}")

    for key in args.models:
        if key not in TRAINERS:
            print(f"Unknown model key: {key}. Choose from {list(TRAINERS.keys())}")
            continue

        name, trainer_fn, output_path = TRAINERS[key]
        model, feats = trainer_fn(train_df, test_df)
        if model is not None:
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(model, output_path)
            size_mb = Path(output_path).stat().st_size / 1024 / 1024
            print(f"  Saved {name} → {output_path} ({size_mb:.2f} MB)")

        if args.walk_forward and model is not None:
            walk_forward(df, feats, trainer_fn, label=name)

    print("\n✅ Training complete!")
