# === File: train_model.py ===

import pandas as pd
import os
import glob
from models import LinearTrafficModel, RandomForestTrafficModel

print("Starting script...")

# === Load and concatenate weather data ===
weather_files = sorted(glob.glob("RawDataFiles/weather_data_vm*.csv"))
if not weather_files:
    raise FileNotFoundError("No weather_data_vm*.csv files found.")

dfs = []
for file in weather_files:
    print(f"Loading {file}")
    df = pd.read_csv(file)
    df['datetime'] = pd.to_datetime(df['date'])
    df['Yr'] = df['datetime'].dt.year
    df['M'] = df['datetime'].dt.month
    df['D'] = df['datetime'].dt.day
    df['HH'] = df['datetime'].dt.hour
    dfs.append(df)

weather_df = pd.concat(dfs, ignore_index=True)
print(f"Weather data concatenated: {weather_df.shape}")

# === Load traffic data ===
traffic_path = "RawDataFiles/automated_traffic_volume_counts.csv"
if not os.path.exists(traffic_path):
    raise FileNotFoundError("Traffic dataset not found.")
traffic_df = pd.read_csv(traffic_path)
traffic_df.rename(columns={'Boro': 'borough'}, inplace=True)

# === Merge datasets ===
for col in ['Yr', 'M', 'D', 'HH', 'borough']:
    if col not in weather_df.columns or col not in traffic_df.columns:
        raise KeyError(f"Missing {col} in one of the dataframes.")
weather_df['HH'] = weather_df['HH'].astype(int)
traffic_df['HH'] = traffic_df['HH'].astype(int)

merged_df = pd.merge(traffic_df, weather_df, on=['Yr', 'M', 'D', 'HH', 'borough'], how='inner')
print(f"Merged dataset shape: {merged_df.shape}")

# === Add derived features ===
merged_df["weekday"] = pd.to_datetime(merged_df["date"]).dt.weekday  # 0 = Monday
merged_df["hour_group"] = merged_df["HH"].apply(lambda h: 1 if 7 <= h <= 9 or 16 <= h <= 18 else 0)

# === Select features ===
features = [
    "Yr", "M", "D", "HH", "weekday", "hour_group",
    "temperature_2m", "precipitation", "cloud_cover_low",
    "snow_depth", "rain", "showers", "snowfall"
]

print("Final features used for training:", features)

# === Drop rows with missing values in important columns ===
print("\n=== Missing values before drop ===")
print(merged_df[features + ["Vol"]].isnull().sum())
print(f"Rows before drop: {merged_df.shape[0]}")
merged_df.dropna(subset=features + ["Vol"], inplace=True)
print(f"Rows after drop: {merged_df.shape[0]}")

# === Save cleaned dataset ===
merged_df.to_csv("RawDataFiles/merged_weather_traffic.csv", index=False)
print("Merged data saved.")

# === Choose and initialize model ===
model = LinearTrafficModel()       # Use linear model helps show which features matter most and the limits of linear

# model = RandomForestTrafficModel()   # Use Random Forest for better accuracy

# === Train and evaluate ===
model.load_csv_data(dataframe=merged_df, target_col="Vol", feature_col=features)
model.split_data()
model.train_model()
model.evaluate()
