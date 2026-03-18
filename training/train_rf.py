import numpy as np
import pandas as pd
import os
import glob
import warnings
import matplotlib.pyplot as plt
from typing import Optional, List
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

warnings.filterwarnings('ignore')

class LinRegressionTemplate:
    def __init__(self):
        # Use Random Forest instead of Linear Regression
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self.is_fitted = False

    def load_csv_data(self, dataframe, target_col: str, feature_col: Optional[List[str]] = None):
        # Extract the target column (what we want to predict)
        self.target_name = target_col
        y = dataframe[target_col]

        # Use provided features or infer from dataframe
        if feature_col is None:
            feature_col = [col for col in dataframe.columns if col != target_col]
        x = dataframe[feature_col]
        self.feature_names = feature_col
        self.x = x
        self.y = y

        print(x.shape)
        print(self.feature_names)
        print(self.target_name)

    def split_data(self, test_size: float = 0.15, random_state: int = 62):
        # Split dataset into training and testing sets
        if self.x is None or self.y is None:
            print('No data to split.')
            return
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state
        )
        print("success")
        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Testing samples: {self.x_test.shape[0]}")

    def train_model(self):
        if self.x_train is None or self.y_train is None:
            print('No training data available.')
            return
        self.model.fit(self.x_train, self.y_train)
        self.is_fitted = True
        print("Model Trained")

    def plot_predictions(self):
        # Visualize actual vs predicted traffic volumes
        if not self.is_fitted:
            print("Model not trained.")
            return
        y_pred = self.model.predict(self.x_test)

        plt.figure(figsize=(12, 5))
        plt.plot(self.y_test.values[:1000], label="Actual", alpha=0.7)
        plt.plot(y_pred[:1000], label="Predicted", alpha=0.7)
        plt.title("Predicted vs Actual Traffic Volume (Sample of 1000)")
        plt.xlabel("Sample Index")
        plt.ylabel("Volume")
        plt.legend()
        plt.tight_layout()
        plt.savefig("prediction_vs_actual.png")
        plt.show()

    def evaluate(self):
        # Evaluate the trained model on train and test sets
        if not self.is_fitted:
            print('Model not trained yet.')
            return
        y_trainPredict = self.model.predict(self.x_train)
        y_testPredict = self.model.predict(self.x_test)

        train_r2 = r2_score(self.y_train, y_trainPredict)
        test_r2 = r2_score(self.y_test, y_testPredict)
        train_mae = mean_absolute_error(self.y_train, y_trainPredict)
        test_mae = mean_absolute_error(self.y_test, y_testPredict)
        train_mse = mean_squared_error(self.y_train, y_trainPredict)
        test_mse = mean_squared_error(self.y_test, y_testPredict)

        print(f"Training R² Score: {train_r2:.4f}")
        print(f"Testing R² Score: {test_r2:.4f}")
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")

        self.plot_predictions()

        # Show feature importances
        if hasattr(self.model, "feature_importances_"):
            print("\nFeature Importances:")
            for name, score in zip(self.feature_names, self.model.feature_importances_):
                print(f"{name}: {score:.4f}")


if __name__ == "__main__":
    print("Starting script...")

    # === Load and concatenate all weather files ===
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

    # === Merge datasets on matching keys ===
    for col in ['Yr', 'M', 'D', 'HH', 'borough']:
        if col not in weather_df.columns or col not in traffic_df.columns:
            raise KeyError(f"Missing {col} in one of the dataframes.")
    weather_df['HH'] = weather_df['HH'].astype(int)
    traffic_df['HH'] = traffic_df['HH'].astype(int)

    merged_df = pd.merge(traffic_df, weather_df, on=['Yr', 'M', 'D', 'HH', 'borough'], how='inner')
    print(f"Merged dataset shape: {merged_df.shape}")

    # === Add derived features (weekday, rush hour) ===
    merged_df["weekday"] = pd.to_datetime(merged_df["date"]).dt.weekday  # 0 = Monday
    merged_df["hour_group"] = merged_df["HH"].apply(lambda h: 1 if 7 <= h <= 9 or 16 <= h <= 18 else 0)

    # === Select features for training ===
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

    # === Train and evaluate model ===
    model = LinRegressionTemplate()
    model.load_csv_data(dataframe=merged_df, target_col="Vol", feature_col=features)
    model.split_data()
    model.train_model()
    model.evaluate()
