# %% [markdown]
# # NYC Traffic Prediction: Hybrid Model (Random Forest + Time-Series)
# **Combines**:
# - Random Forest for short-term congestion (0-6 hours)
# - LSTM/Prophet for long-term trends (daily/weekly patterns)

# %% [markdown]
# ## 1. Setup
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
from prophet import Prophet
%matplotlib inline

# %% [markdown]
# ## 2. Data Preparation (Time-Series Format)
# %%
# Generate time-series traffic data (hourly for 6 months)
date_rng = pd.date_range(start='2023-01-01', end='2023-06-30', freq='H')
traffic_data = {
    'timestamp': date_rng,
    'congestion': np.sin(np.arange(len(date_rng)) * 5 + 5 + np.random.normal(0, 1, len(date_rng)),  # Synthetic pattern
    'hour': date_rng.hour,
    'is_weekend': (date_rng.weekday >= 5).astype(int)
}
df_ts = pd.DataFrame(traffic_data).set_index('timestamp')

# Feature engineering
df_ts['lag_1h'] = df_ts['congestion'].shift(1)  # Previous hour's congestion
df_ts = df_ts.dropna()

# %% [markdown]
# ## 3. Short-Term Prediction (Random Forest)
# %%
# Same as previous notebook but with time features
X = df_ts[['hour', 'is_weekend', 'lag_1h']]
y = df_ts['congestion']

# Train-test split (temporal)
split_idx = int(len(df_ts)*0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

# Train RF
rf_model = RandomForestRegressor(n_estimators=100)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)

print(f"RF MAE: {mean_absolute_error(y_test, rf_pred):.2f}")

# %% [markdown]
# ## 4. Long-Term Prediction (LSTM)
# %%
# Prepare LSTM data
def create_dataset(data, lookback=24):
    X, y = [], []
    for i in range(len(data)-lookback):
        X.append(data[i:(i+lookback)])
        y.append(data[i+lookback])
    return np.array(X), np.array(y)

lstm_data = df_ts['congestion'].values.reshape(-1, 1)
X_lstm, y_lstm = create_dataset(lstm_data)

# Build LSTM model
lstm_model = Sequential([
    LSTM(50, input_shape=(24, 1)),
    Dense(1)
])
lstm_model.compile(optimizer='adam', loss='mse')
lstm_model.fit(X_lstm, y_lstm, epochs=5, batch_size=32)

# %% [markdown]
# ## 5. Very Long-Term Prediction (Prophet)
# %%
# Prepare Prophet data
df_prophet = df_ts.reset_index()[['timestamp', 'congestion']].rename(
    columns={'timestamp': 'ds', 'congestion': 'y'})

# Train Prophet
prophet_model = Prophet(seasonality_mode='multiplicative')
prophet_model.add_country_holidays(country_name='US')
prophet_model.fit(df_prophet)

# Make future forecast
future = prophet_model.make_future_dataframe(periods=720, freq='H')  # 30 days
forecast = prophet_model.predict(future)

# %% [markdown]
# ## 6. Hybrid Prediction System
# %%
def predict_congestion(timestamp, current_congestion):
    """Combines all models for predictions at any time horizon"""
    
    # Feature preparation
    hour = timestamp.hour
    is_weekend = (timestamp.weekday() >= 5)
    
    # Short-term (RF)
    rf_input = [[hour, is_weekend, current_congestion]]
    rf_pred = rf_model.predict(rf_input)[0]
    
    # Long-term (LSTM + Prophet)
    lstm_pred = ...  # (Implementation depends on production setup)
    prophet_pred = forecast[forecast['ds'] == timestamp]['yhat'].values[0]
    
    return {
        '0-6_hours': rf_pred,
        '7-24_hours': lstm_pred,
        '1-30_days': prophet_pred
    }

# Example usage
pred = predict_congestion(
    timestamp=pd.to_datetime('2023-07-01 17:00:00'),
    current_congestion=6.2
)
print(pred)

# %% [markdown]
# ## 7. Visualization
# %%
fig, ax = plt.subplots(3, 1, figsize=(15, 12))

# RF Results
ax[0].plot(y_test.values[:100], label='Actual')
ax[0].plot(rf_pred[:100], label='RF Predicted')
ax[0].set_title('Random Forest (Next Hour Prediction)')

# Prophet Forecast
prophet_model.plot(forecast, ax=ax[1])
ax[1].set_title('Prophet (30-Day Trend)')

# Feature Importance
importances = rf_model.feature_importances_
ax[2].bar(X.columns, importances)
ax[2].set_title('RF Feature Importance')

plt.tight_layout()
plt.show()