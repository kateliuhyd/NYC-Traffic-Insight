# save_models.py
"""
Train and serialize three traffic-volume models:
 1) HistGradientBoostingRegressor
 2) RandomForestRegressor
 3) SegmentedModel wrapper
Uses SegmentedModeling.load_and_prepare_data to load and preprocess.
"""
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from SegmentedModeling import load_and_prepare_data, SegmentedModel


def main():
    # Load & preprocess full dataset
    df_clean, feature_cols = load_and_prepare_data()
    # 80/20 time-based split
    split_idx = int(len(df_clean) * 0.8)
    train_df = df_clean.iloc[:split_idx]
    X_train = train_df[feature_cols]
    y_train = train_df['Vol_log']

    # 1) Train & save HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        early_stopping=True,
        random_state=42
    )
    hgb.fit(X_train, y_train)
    joblib.dump(hgb, "hgb_model.joblib")
    print("Saved HistGradientBoostingRegressor to hgb_model.joblib")

    # 2) Train & save RandomForestRegressor
    rf = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    joblib.dump(rf, "rf_model.joblib")
    print("Saved RandomForestRegressor to rf_model.joblib")

    # 3) Train & save SegmentedModel
    seg = SegmentedModel()
    # SegmentedModel.fit splits internally by time and segments
    seg.fit(df_clean, feature_cols)
    joblib.dump(seg, "segmented_model.joblib")
    print("Saved SegmentedModel to segmented_model.joblib")


if __name__ == "__main__":
    main()
