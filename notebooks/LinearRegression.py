import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Optional, List
import warnings
warnings.filterwarnings('ignore')


class LinRegressionTemplate:
    def __init__(self):
        self.model = LinearRegression()
        
        #data variables
        self.x_train = None #input training
        self.x_test = None #input testing
        self.y_train = None # training target
        self.y_test = None #  testing target
        self.feature_names = None # name of the input variables
        self.target_name = None #name of what we are predicting
        #is madel trained or not?
        self.is_fitted = False

    def load_csv_data(self, dataframe, target_col: str, feature_col: Optional[List[str]] = None):
        # getting the target variable
        self.target_name = target_col
        y = dataframe[target_col] #column we want to predict

        # gets the feature variables
        if feature_col is None:
            # If no specific features are specified, use all columns except the target
            feature_col = [col for col in dataframe.columns if col != target_col]
        x = dataframe[feature_col] #gets feature variables
        self.feature_names = feature_col
        x = dataframe[feature_col]

        #store data for other functions
        self.x = x
        self.y = y
        print(x.shape)
        print(self.feature_names)
        print(self.target_name)

    def split_data(self, test_size: float = 0.15, random_state: int = 62):
        #dividing training set and testing sets into two groups
        if self.x is None or self.y is None:
            print('No data to split.')
            return

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x,self.y,test_size=test_size,random_state=random_state)
        print("success")
        print(self.x_train.shape[0])
        print(self.x_test.shape[0])

    def train_model(self):
        if self.x_train is None or self.y_train is None:
            print('No training data, split the data')
            return

        self.model.fit(self.x_train, self.y_train)
        #model trained
        self.is_fitted = True
        print("Model Trained")

    def evaluate(self):
        if not self.is_fitted:
            print('Model not trained')
            return

        #predictions on both groups of data
        y_trainPredict = self.model.predict(self.x_train)
        y_testPredict = self.model.predict(self.x_test)


        #Evaluations on the results (r^2, MAE, MSE)

        #r^2 (1.0 = perfect, 0.0 bad)
        train_r2 = r2_score(self.y_train, y_trainPredict)
        test_r2 = r2_score(self.y_test, y_testPredict)
        #MAE (Lower is better)
        train_mae = mean_absolute_error(self.y_train, y_trainPredict)
        test_mae = mean_absolute_error(self.y_test, y_testPredict)
        #MSE (Lower is better)
        train_mse = mean_squared_error(self.y_train, y_trainPredict)
        test_mse = mean_squared_error(self.y_test, y_testPredict)

        print(f"Training R² Score: {train_r2:.4f}")  # How well it fits training data
        print(f"Testing R² Score: {test_r2:.4f}")    # How well it performs on new data
        print(f"Training MSE: {train_mse:.4f}")
        print(f"Testing MSE: {test_mse:.4f}")
        print(f"Training MAE: {train_mae:.4f}")
        print(f"Testing MAE: {test_mae:.4f}")
