"""
experimental/train_model.py
Enhanced RandomForest model training for NYC traffic volume prediction.
Expected Test R² ~ 0.9261 on log-transformed target.
"""

import os
import glob
import pandas as pd
import numpy as np
import holidays
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split


def load_and_prepare():
    # 1) Load weather data
    pattern = "RawDataFiles/weather_data_vm*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No weather files match {pattern}")
    weather_dfs = []
    for f in files:
        print(f"Loading {f}")
        for chunk in pd.read_csv(f, parse_dates=['date'], chunksize=200_000):
            chunk['Yr'] = chunk['date'].dt.year
            chunk['M']  = chunk['date'].dt.month
            chunk['D']  = chunk['date'].dt.day
            chunk['HH'] = chunk['date'].dt.hour
            weather_dfs.append(
                chunk[['Yr','M','D','HH','precipitation','temperature_2m','cloud_cover_low','snow_depth']]
            )
    weather_df = pd.concat(weather_dfs, ignore_index=True)
    print(f"Weather data concatenated: {weather_df.shape}")

    # 2) Load traffic data
    traffic_path = "RawDataFiles/automated_traffic_volume_counts.csv"
    if not os.path.exists(traffic_path):
        raise FileNotFoundError(f"Traffic data not found at {traffic_path}")
    traffic_df = pd.read_csv(traffic_path)
    traffic_df.rename(columns={'Boro':'borough'}, inplace=True)
    print(f"Traffic data loaded: {traffic_df.shape}")

    # 3) Merge
    for col in ['Yr','M','D','HH']:
        weather_df[col] = weather_df[col].astype(int)
        traffic_df[col] = traffic_df[col].astype(int)
    merged = pd.merge(
        traffic_df, weather_df,
        on=['Yr','M','D','HH'], how='inner'
    )
    print(f"Merged dataset shape: {merged.shape}")

    # 4) Feature engineering
    merged['hour_sin']  = np.sin(2 * np.pi * merged['HH'] / 24)
    merged['hour_cos']  = np.cos(2 * np.pi * merged['HH'] / 24)
    merged['weekday']   = pd.to_datetime(merged['date']).dt.weekday
    merged['wd_sin']    = np.sin(2 * np.pi * merged['weekday'] / 7)
    merged['wd_cos']    = np.cos(2 * np.pi * merged['weekday'] / 7)
    merged['month_sin'] = np.sin(2 * np.pi * merged['M'] / 12)
    merged['month_cos'] = np.cos(2 * np.pi * merged['M'] / 12)

    # one-hot borough
    borough_dummies = pd.get_dummies(merged['borough'], prefix='b')
    merged = pd.concat([merged, borough_dummies], axis=1)

    # holiday flag
    us_holidays = holidays.US()
    merged['is_holiday'] = merged['date'].dt.date.isin(us_holidays).astype(int)

    # lag features
    merged = merged.sort_values(['borough', 'date'])
    for lag in [1, 24, 168]:
        merged[f'vol_lag_{lag}'] = merged.groupby('borough')['Vol'].shift(lag)

    # log-transform target
    merged['Vol_log'] = np.log1p(merged['Vol'])

    # drop NAs
    needed = ['Vol_log'] + [f'vol_lag_{lag}' for lag in (1,24,168)]
    merged.dropna(subset=needed, inplace=True)
    merged.reset_index(drop=True, inplace=True)

    # define features
    base_feats = ['hour_sin','hour_cos','wd_sin','wd_cos','month_sin','month_cos','is_holiday']
    boro_feats = [c for c in merged.columns if c.startswith('b_')]
    lag_feats  = [f'vol_lag_{lag}' for lag in (1,24,168)]
    features = base_feats + boro_feats + lag_feats

    return merged, features


def main():
    df, features = load_and_prepare()

    # time-based split
    df['dt_full'] = pd.to_datetime(
        df[['Yr','M','D','HH']].astype(str).agg('-'.join, axis=1),
        format='%Y-%m-%d-%H'
    )
    df.sort_values('dt_full', inplace=True)
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df  = df.iloc[split_idx:].reset_index(drop=True)
    X_train, y_train = train_df[features], train_df['Vol_log']
    X_test,  y_test  = test_df[features],  test_df['Vol_log']

    # train Random Forest
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    # evaluate
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    print(f"Training R² Score: {r2_score(y_train, y_train_pred):.4f}")
    print(f"Testing R² Score:  {r2_score(y_test, y_test_pred):.4f}")
    print(f"Training MSE:       {mean_squared_error(y_train, y_train_pred):.4f}")
    print(f"Testing MSE:        {mean_squared_error(y_test, y_test_pred):.4f}")
    print(f"Training MAE:       {mean_absolute_error(y_train, y_train_pred):.4f}")
    print(f"Testing MAE:        {mean_absolute_error(y_test, y_test_pred):.4f}")

    # feature importances
    print("\nFeature Importances:")
    for name, score in sorted(
        zip(features, model.feature_importances_), key=lambda x: x[1], reverse=True
    ):
        print(f"{name}: {score:.4f}")


if __name__ == '__main__':
    main()
