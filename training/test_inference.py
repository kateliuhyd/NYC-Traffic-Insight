# test_inference.py
"""
Sanity check for trained traffic-volume models:
- HistGradientBoostingRegressor (hgb_model.joblib)
- RandomForestRegressor       (rf_model.joblib)
- SegmentedModel             (segmented_model.joblib)

Usage:
    pip install pandas numpy joblib
    python test_inference.py
"""
import joblib
import pandas as pd
import numpy as np

# Features used during HGB and RF training
FEATURE_COLS = [
    'hour_sin','hour_cos','wd_sin','wd_cos',
    'month_sin','month_cos',
    'vol_lag_1','vol_roll_3h','vol_roll_24h'
]
# Event flags used during segmented model training
EVENT_COLS = ['is_holiday', 'heavy_snow']


def load_models():
    """Load serialized models from disk."""
    hgb = joblib.load('hgb_model.joblib')
    rf  = joblib.load('rf_model.joblib')
    seg = joblib.load('segmented_model.joblib')
    return hgb, rf, seg


def make_sample():
    """Create a dummy sample matching training schema."""
    sample = {
        'hour_sin':    0.0,
        'hour_cos':    1.0,
        'wd_sin':      0.0,
        'wd_cos':      1.0,
        'month_sin':   0.5,
        'month_cos':   0.866,
        'vol_lag_1':   100,
        'vol_roll_3h': 110,
        'vol_roll_24h':115,
        'is_holiday':   0,
        'heavy_snow':   0
    }
    # DataFrame for HGB and RF
    df_feat = pd.DataFrame([sample])[FEATURE_COLS]
    # DataFrame for SegmentedModel (requires event flags)
    df_full = pd.DataFrame([sample])
    return df_feat, df_full


def main():
    # Load models
    hgb_model, rf_model, seg_model = load_models()

    # Prepare sample inputs
    X_feat, X_full = make_sample()

    # 1) HistGradientBoostingRegressor
    log_hgb = hgb_model.predict(X_feat)
    raw_hgb = np.expm1(log_hgb)
    print('=== HGB Model ===')
    print(f'Log-volume prediction: {log_hgb[0]:.4f}')
    print(f'Raw-volume prediction: {raw_hgb[0]:.2f}\n')

    # 2) RandomForestRegressor
    log_rf = rf_model.predict(X_feat)
    raw_rf = np.expm1(log_rf)
    print('=== Random Forest Model ===')
    print(f'Log-volume prediction: {log_rf[0]:.4f}')
    print(f'Raw-volume prediction: {raw_rf[0]:.2f}\n')

    # 3) SegmentedModel
    raw_seg = seg_model.predict(X_full)
    print('=== Segmented Model ===')
    print(f'Raw-volume prediction: {raw_seg[0]:.2f}')


if __name__ == '__main__':
    main()
