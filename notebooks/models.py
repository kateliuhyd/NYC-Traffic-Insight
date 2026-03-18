# === File: models.py ===

import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


class BaseTrafficModel:
    def __init__(self):
        self.model = None
        self.x_train = None
        self.x_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = None
        self.target_name = None
        self.is_fitted = False

    def load_csv_data(self, dataframe, target_col, feature_col):
        self.target_name = target_col
        self.feature_names = feature_col
        self.x = dataframe[feature_col]
        self.y = dataframe[target_col]
        print(self.x.shape)
        print(self.feature_names)
        print(self.target_name)

    def split_data(self, test_size=0.15, random_state=42):
        from sklearn.model_selection import train_test_split
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=test_size, random_state=random_state)
        print("success")
        print(f"Training samples: {self.x_train.shape[0]}")
        print(f"Testing samples: {self.x_test.shape[0]}")

    def plot_predictions(self, y_pred):
        plt.figure(figsize=(12, 5))
        plt.plot(self.y_test.values[:1000], label="Actual", alpha=0.7)
        plt.plot(y_pred[:1000], label="Predicted", alpha=0.7)
        plt.title(f"{self.__class__.__name__} Prediction vs Actual")
        plt.xlabel("Sample Index")
        plt.ylabel("Volume")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.__class__.__name__}_prediction.png")
        plt.show()

    def evaluate(self):
        if not self.is_fitted:
            print('Model not trained.')
            return
        y_train_pred = self.model.predict(self.x_train)
        y_test_pred = self.model.predict(self.x_test)

        print(f"Training R² Score: {r2_score(self.y_train, y_train_pred):.4f}")
        print(f"Testing R² Score: {r2_score(self.y_test, y_test_pred):.4f}")
        print(f"Training MSE: {mean_squared_error(self.y_train, y_train_pred):.4f}")
        print(f"Testing MSE: {mean_squared_error(self.y_test, y_test_pred):.4f}")
        print(f"Training MAE: {mean_absolute_error(self.y_train, y_train_pred):.4f}")
        print(f"Testing MAE: {mean_absolute_error(self.y_test, y_test_pred):.4f}")
        self.plot_predictions(y_test_pred)


class LinearTrafficModel(BaseTrafficModel):
    def __init__(self):
        super().__init__()
        self.model = LinearRegression()
        print("Initialized LinearRegression Model")

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        self.is_fitted = True
        print("Linear model trained")


class RandomForestTrafficModel(BaseTrafficModel):
    def __init__(self):
        super().__init__()
        self.model = RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
        print("Initialized RandomForestRegressor Model")

    def train_model(self):
        self.model.fit(self.x_train, self.y_train)
        self.is_fitted = True
        print("Random Forest model trained")

        # Show feature importances
        print("\nFeature Importances:")
        for name, score in zip(self.feature_names, self.model.feature_importances_):
            print(f"{name}: {score:.4f}")
