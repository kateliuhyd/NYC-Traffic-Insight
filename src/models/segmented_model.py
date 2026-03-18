"""
SegmentedModel — trains separate HistGradientBoosting estimators for
normal-traffic periods vs. event periods (holidays / heavy snow).

This is the **single source of truth** for the class.  Both
``training/train_segmented.py`` (offline) and
``src/model_loader.py`` (serving via joblib) reference this module.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)
from sklearn.inspection import permutation_importance


class SegmentedModel:
    """Two-regime traffic-volume model (normal vs. event)."""

    def __init__(
        self,
        normal_params: dict | None = None,
        event_params: dict | None = None,
        oversample_factor: int = 5,
    ):
        self.normal_params = normal_params or {
            "max_iter": 200,
            "learning_rate": 0.1,
            "max_depth": 6,
            "early_stopping": True,
            "random_state": 42,
        }
        self.event_params = event_params or self.normal_params.copy()
        self.oversample_factor = oversample_factor
        self.model_normal = None
        self.model_event = None
        self.features: list[str] = []
        self.train_df: pd.DataFrame | None = None
        self.test_df: pd.DataFrame | None = None

    # ── Training ──────────────────────────────────────────────────

    def fit(
        self,
        df: pd.DataFrame,
        features: list[str],
        target_col: str = "Vol_log",
        event_cols: tuple[str, str] = ("is_holiday", "heavy_snow"),
        train_frac: float = 0.8,
    ) -> "SegmentedModel":
        self.features = features
        df_clean = df.dropna(subset=features + [target_col]).reset_index(drop=True)

        split_idx = int(len(df_clean) * train_frac)
        self.train_df = df_clean.iloc[:split_idx].copy()
        self.test_df = df_clean.iloc[split_idx:].copy()

        mask_event = (
            self.train_df[event_cols[0]].astype(bool)
            | self.train_df[event_cols[1]].astype(bool)
        )

        # Normal-period model
        self.model_normal = HistGradientBoostingRegressor(**self.normal_params)
        self.model_normal.fit(
            self.train_df.loc[~mask_event, features],
            self.train_df.loc[~mask_event, target_col],
        )

        # Event-period model (with optional oversampling)
        evt_df = self.train_df.loc[mask_event].copy()
        if len(evt_df) < len(self.train_df) * 0.1:
            evt_df = pd.concat(
                [evt_df] * self.oversample_factor, ignore_index=True
            ).sample(frac=1, random_state=42)

        self.model_event = HistGradientBoostingRegressor(**self.event_params)
        self.model_event.fit(evt_df[features], evt_df[target_col])

        return self

    # ── Inference ─────────────────────────────────────────────────

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Return **raw** (un-log-transformed) volume predictions."""
        mask_event = (
            df["is_holiday"].astype(bool) | df["heavy_snow"].astype(bool)
        )
        X = df[self.features]
        pred_log = np.where(
            mask_event,
            self.model_event.predict(X),
            self.model_normal.predict(X),
        )
        return np.expm1(pred_log)

    # ── Evaluation ────────────────────────────────────────────────

    def evaluate(self) -> None:
        """Print metrics for segmented vs. baseline (normal-only) models."""
        if self.test_df is None:
            raise ValueError("Must call fit() before evaluate().")

        y_true = self.test_df["Vol"]
        y_pred = self.predict(self.test_df)
        nz = y_true > 0

        print("Segmented Model Evaluation:")
        print(f"  Test R²   (raw): {r2_score(y_true, y_pred):.4f}")
        print(f"  Test MAE  (raw): {mean_absolute_error(y_true, y_pred):.2f}")
        print(
            f"  Test MAPE (raw): "
            f"{mean_absolute_percentage_error(y_true[nz], y_pred[nz]) * 100:.2f}%"
        )

        # Baseline comparison
        y_base = np.expm1(
            self.model_normal.predict(self.test_df[self.features])
        )
        print("Baseline (Normal-Only) Evaluation:")
        print(f"  Test R²   (raw): {r2_score(y_true, y_base):.4f}")
        print(f"  Test MAE  (raw): {mean_absolute_error(y_true, y_base):.2f}")
        print(
            f"  Test MAPE (raw): "
            f"{mean_absolute_percentage_error(y_true[nz], y_base[nz]) * 100:.2f}%"
        )

        # Permutation importances (non-event subset)
        mask_ne = ~(
            self.test_df["is_holiday"].astype(bool)
            | self.test_df["heavy_snow"].astype(bool)
        )
        perm = permutation_importance(
            self.model_normal,
            self.test_df.loc[mask_ne, self.features],
            np.log1p(self.test_df.loc[mask_ne, "Vol"]),
            n_repeats=5,
            random_state=42,
            n_jobs=-1,
        )
        feat_imp = sorted(
            zip(self.features, perm.importances_mean),
            key=lambda x: x[1],
            reverse=True,
        )
        print("\nFeature importances (permutation, normal-period):")
        for name, imp in feat_imp:
            print(f"  {name}: {imp:.6f}")

    # ── Plotting ──────────────────────────────────────────────────

    def plot(self, n_samples: int = 1000) -> None:
        """Plot actual vs. segmented vs. baseline (first *n_samples*)."""
        import matplotlib.pyplot as plt

        if self.test_df is None:
            raise ValueError("Must call fit() before plot().")

        y_true = self.test_df["Vol"].values
        y_pred = self.predict(self.test_df)
        y_base = np.expm1(
            self.model_normal.predict(self.test_df[self.features])
        )
        N = min(n_samples, len(y_true))

        plt.figure(figsize=(12, 5))
        plt.plot(y_true[:N], label="Actual", alpha=0.7)
        plt.plot(y_pred[:N], label="Segmented", alpha=0.7)
        plt.plot(y_base[:N], label="Baseline", alpha=0.7)
        plt.title(f"Actual vs Segmented vs Baseline (first {N} samples)")
        plt.xlabel("Sample Index")
        plt.ylabel("Traffic Volume")
        plt.legend()
        plt.tight_layout()
        plt.show()
