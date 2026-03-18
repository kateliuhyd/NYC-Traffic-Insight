import pandas as pd
import numpy as np
import holidays

# Change to something dynamic
df = pd.read_csv(r"C:\Users\nokon\Downloads\merged_traffic_data.csv")

df["datetime"] = pd.to_datetime(df[["Yr", "M", "D", "HH"]])
df["month"] = df["datetime"].dt.month
df["quarter"] = df["datetime"].dt.quarter
df["dayofweek"] = df["datetime"].dt.dayofweek