# Credit: Adapted from https://www.kaggle.com/code/owczar/gam-model-for-time-series-forecasting/notebook
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("../ETT-small/ETTh1.csv")

# # Explore dataset
# print(f"Head:\n {df.head()}")
# print(f"\nData Types:\n {df.dtypes}")
print(f"\nShape:\n {df.shape}")
# print("\n", df.isna().sum())  # No missing values!


# --- Feature engineering --- #
# Convert date column to datetime and set as index
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

df['time'] = np.arange(len(df))  # Create a time index (0, 1, 2, ..., n-1)

# Monthly seasonality
month = df.index.month
df['month_sin'] = np.sin(2*np.pi*month/12)
df['month_cos'] = np.cos(2*np.pi*month/12)

# Hourly seasonality
df['hour'] = df.index.hour
df['hour_sin'] = np.sin(2*np.pi*df['hour']/24)
df['hour_cos'] = np.cos(2*np.pi*df['hour']/24)

# Daily seasonality
df['dow'] = df.index.dayofweek   # 0–6
df['dayOfWeek_sin'] = np.sin(2*np.pi*df['dow']/7)
df['dayOfWeek_cos'] = np.cos(2*np.pi*df['dow']/7)


# --- Define target and features --- #
from pygam import GAM, s

# Define target and features:
# y (Oil Temperature column)
y = df['OT']
# X (feature columns)
X = df[[
    'time', 
    'month_sin', 'month_cos', 
    'hour_sin', 'hour_cos', 
    'dayOfWeek_sin', 'dayOfWeek_cos',
    'HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL'
    ]] 


# --- Fit GAM with multiple smooths --- #
gam = GAM(
    s(0, n_splines=50) +  # smooth function of time
    s(1, n_splines=10) +  # smooth function of month_sin
    s(2, n_splines=10) +  # smooth function of month_cos
    s(3, n_splines=15) +  # smooth function of hour_sin
    s(4, n_splines=15) +  # smooth function of hour_cos
    s(5, n_splines=10) +  # smooth function of dayOfWeek_sin
    s(6, n_splines=10) +  # smooth function of dayOfWeek_cos
    s(7, n_splines=20) +   # HUFL
    s(8, n_splines=20) +   # HULL
    s(9, n_splines=20) +   # MUFL
    s(10, n_splines=20) +  # MULL
    s(11, n_splines=20) +  # LUFL
    s(12, n_splines=20)    # LULL
)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(0.8 * len(df))
X_train = X.iloc[:train_size]  # First 80% of rows for training
X_test = X.iloc[train_size:]  # Last 20% of rows for testing

y_train = y.iloc[:train_size]
y_test = y.iloc[train_size:]

# Fit (train) the GAM
gam.fit(X_train, y_train)


# --- Calculate Mean Squared Error and Mean Absolute Error --- #
from sklearn.metrics import mean_squared_error, mean_absolute_error

y_pred = gam.predict(X_test)

gam_mse = mean_squared_error(y_test, y_pred)
gam_mae = mean_absolute_error(y_test, y_pred)

print(f'Mean Squared Error: {gam_mse}')
print(f'Mean Absolute Error: {gam_mae}')
# Mean Squared Error: 221.8722436453739
# Mean Absolute Error: 13.27899125219863


# --- Plot Predicted values vs Test (actual values) --- #
from statsmodels.nonparametric.smoothers_lowess import lowess

# Convert predictions to a Series indexed by X_test time index
pred_series = pd.Series(y_pred, index=X_test.index)

# LOWESS smoothing
# frac controls smoothness: smaller = less smooth, larger = more smooth
lowess_out = lowess(
    endog=pred_series.values,
    exog=np.arange(len(pred_series)),
    frac=0.03
)
pred_lowess = pd.Series(lowess_out[:, 1], index=X_test.index)

# Plot
plt.figure(figsize=(12, 6))

# Train and test data
plt.plot(X_train['time'], y_train, label="Train")
plt.plot(X_test['time'], y_test, label="Test", alpha=0.7)

# Raw predictions (faint)
# plt.plot(X_test['time'], pred_series, color='green', alpha=0.3, label="Predicted (raw)")

# Smoothed predictions (bold)
plt.plot(X_test['time'], pred_lowess, color='green', linewidth=2.0, label="Predicted (LOWESS smooth)")

plt.legend()
plt.show()


# ----------------------------------------------------------------- #

# --- Prophet model for comparison --- #
from prophet import Prophet

# Prepare data for Prophet
# Prophet requires columns to be named 'ds' and 'y'
df_prophet = df.reset_index()[['date', 'OT']].rename(columns={'date': 'ds', 'OT': 'y'}) 

# Split into train (80%) and test (20%) sets
train_set = df_prophet.iloc[:train_size]
test_set = df_prophet.iloc[train_size:]

# Define Prophet model without default seasonalities
prophet = Prophet(
    changepoint_prior_scale=0.02,   # smoother trend
    n_changepoints=100,             # more knots (similar to splines)
    weekly_seasonality=False,
    daily_seasonality=False,
    yearly_seasonality=False
)

# Add custom seasonalities
prophet.add_seasonality(name='yearly', period=365.25, fourier_order=10)
prophet.add_seasonality(name='monthly', period=30.5, fourier_order=5)
prophet.add_seasonality(name='weekly', period=7, fourier_order=6)
prophet.add_seasonality(name='daily', period=24, fourier_order=8)

# Train the model
prophet.fit(train_set)

# Create an hourly future dataframe and make predicitons
future = prophet.make_future_dataframe(periods=len(test_set), freq="h")
forecast = prophet.predict(future)

# Evaluate Prophet model
actuals = test_set['y']
predictions = forecast['yhat'].iloc[-len(test_set):]

prophet_mse = mean_squared_error(actuals, predictions)
prophet_mae = mean_absolute_error(actuals, predictions)

print(f'\nMean Squared Error: {prophet_mse}')
print(f'Mean Absolute Error: {prophet_mae}\n')

# Plot Predicted values vs Test (actual values) for Prophet
plt.figure(figsize=(12, 6))
plt.plot(train_set['ds'], train_set['y'], label="Train")
plt.plot(test_set['ds'], test_set['y'], label="Test")
plt.plot(test_set['ds'], predictions, label="Predicted")
plt.legend()
plt.show()

# --- Summary of comparison results between my GAM and Prophet --- #
results = pd.DataFrame(
    {"GAM": [gam_mse, gam_mae], "Prophet": [prophet_mse, prophet_mae]},
    index=["MSE", "MAE"]
)
print("Comparison of GAM and Prophet models:")
print(results)
