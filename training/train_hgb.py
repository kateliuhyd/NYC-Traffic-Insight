# experimental/train_hgb_model.py
"""
Train and evaluate a HistGradientBoostingRegressor for NYC traffic volume prediction.
Expected Test R² around 0.8889 on raw volume.
"""
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.inspection import permutation_importance


def load_and_merge_data(
    weather_pattern: str = "RawDataFiles/weather_data_vm*.csv",
    traffic_path: str = "RawDataFiles/automated_traffic_volume_counts.csv"
) -> pd.DataFrame:
    # 1) Load weather in chunks
    weather_files = sorted(glob.glob(weather_pattern))
    if not weather_files:
        raise FileNotFoundError(f"No weather files match {weather_pattern}")
    weather_list = []
    usecols_weather = ['date','precipitation','temperature_2m','cloud_cover_low','snow_depth']
    for file in weather_files:
        for chunk in pd.read_csv(
            file,
            usecols=usecols_weather,
            parse_dates=['date'],
            chunksize=200_000
        ):
            chunk['Yr'] = chunk['date'].dt.year
            chunk['M']  = chunk['date'].dt.month
            chunk['D']  = chunk['date'].dt.day
            chunk['HH'] = chunk['date'].dt.hour
            weather_list.append(chunk)
    weather_df = pd.concat(weather_list, ignore_index=True)

    # 2) Load traffic data
    traffic_df = pd.read_csv(
        traffic_path,
        usecols=['Yr','M','D','HH','Boro','Vol']
    )
    traffic_df.rename(columns={'Boro':'borough'}, inplace=True)

    # 3) Merge on keys
    merged = pd.merge(
        traffic_df,
        weather_df,
        on=['Yr','M','D','HH'],
        how='inner',
        copy=False
    )
    merged.sort_values('date', inplace=True)
    merged.reset_index(drop=True, inplace=True)
    return merged


def preprocess(df: pd.DataFrame) -> (pd.DataFrame, list):
    # 4) Feature engineering
    df['hour_sin']  = np.sin(2 * np.pi * df['HH'] / 24)
    df['hour_cos']  = np.cos(2 * np.pi * df['HH'] / 24)
    df['weekday']   = df['date'].dt.weekday
    df['wd_sin']    = np.sin(2 * np.pi * df['weekday'] / 7)
    df['wd_cos']    = np.cos(2 * np.pi * df['weekday'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['M'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['M'] / 12)

    # 5) Lag features
    for lag in [1,24,168]:
        df[f'vol_lag_{lag}'] = df.groupby('borough')['Vol'].shift(lag)

    # 6) Rolling-window features
    df['vol_roll_3h'] = (
        df.groupby('borough')['Vol']
          .rolling(window=3, min_periods=1)
          .mean()
          .shift(1)
          .reset_index(level=0, drop=True)
    )
    df['vol_roll_24h'] = (
        df.groupby('borough')['Vol']
          .rolling(window=24, min_periods=1)
          .mean()
          .shift(1)
          .reset_index(level=0, drop=True)
    )

    # 7) Log-transform target
    df['Vol_log'] = np.log1p(df['Vol'])

    # 8) Drop NaNs
    feature_cols = [
        'hour_sin','hour_cos','wd_sin','wd_cos',
        'month_sin','month_cos',
        'vol_lag_1','vol_lag_24','vol_lag_168',
        'vol_roll_3h','vol_roll_24h'
    ]
    df_clean = df.dropna(subset=feature_cols + ['Vol_log']).reset_index(drop=True)
    return df_clean, feature_cols


def main():
    # load and preprocess
    df = load_and_merge_data()
    df_clean, feature_cols = preprocess(df)

    # 9) Train/test split by time
    df_clean.sort_values('date', inplace=True)
    split_idx = int(len(df_clean) * 0.8)
    train = df_clean.iloc[:split_idx]
    test  = df_clean.iloc[split_idx:]

    X_train, y_train = train[feature_cols], train['Vol_log']
    X_test,  y_test  = test[feature_cols],  test['Vol_log']

    # 10) Train HistGradientBoostingRegressor
    hgb = HistGradientBoostingRegressor(
        max_iter=200,
        learning_rate=0.1,
        max_depth=6,
        early_stopping=True,
        random_state=42
    )
    hgb.fit(X_train, y_train)

    # 11) Permutation importances
    print("Computing permutation importances...")
    perm = permutation_importance(
        hgb, X_test, y_test,
        n_repeats=5, random_state=42, n_jobs=-1
    )
    importances = perm.importances_mean
    feat_imp = sorted(zip(feature_cols, importances), key=lambda x: x[1], reverse=True)
    print("Feature importances (permutation):")
    for name, imp in feat_imp:
        print(f"{name}: {imp:.6f}")

    # 12) Evaluate on log scale
    y_train_pred = hgb.predict(X_train)
    y_test_pred  = hgb.predict(X_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_train_pred))
    rmse_test  = np.sqrt(mean_squared_error(y_test,  y_test_pred))

    print("\nOptimized HistGradientBoosting on Log-Transformed Target:")
    print(f"Train R²: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Test  R²: {r2_score(y_test,  y_test_pred):.4f}")
    print(f"Train RMSE: {rmse_train:.4f}")
    print(f"Test  RMSE: {rmse_test:.4f}")
    print(f"Train MAE:  {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"Test  MAE:   {mean_absolute_error(y_test,  y_test_pred):.4f}")

    # 13) Evaluate on raw scale
    train_raw = np.expm1(y_train_pred)
    test_raw  = np.expm1(y_test_pred)
    print(f"Test R² (raw):  {r2_score(test['Vol'], test_raw):.4f}")
    print(f"Test MAE (raw): {mean_absolute_error(test['Vol'], test_raw):.2f}")

    # 14) Plot actual vs predicted
    N = 1000
    plt.figure(figsize=(12, 5))
    plt.plot(test['Vol'].values[:N], label='Actual', alpha=0.7)
    plt.plot(test_raw[:N], label='Predicted', alpha=0.7)
    plt.title('Actual vs Predicted Traffic Volume (first 1000 samples)')
    plt.xlabel('Sample Index')
    plt.ylabel('Volume')
    plt.legend()
    plt.tight_layout()
    plt.show()

    # 15) Plot importances
    names, imps = zip(*feat_imp)
    plt.figure(figsize=(10, 6))
    plt.bar(names, imps)
    plt.xticks(rotation=45, ha='right')
    plt.title('Permutation Feature Importances')
    plt.ylabel('Importance')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
