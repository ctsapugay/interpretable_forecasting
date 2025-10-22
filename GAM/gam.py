import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pygam import LinearGAM, s, f

# Load dataset
df = pd.read_csv("../ETT-small/ETTh1.csv")
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values('date')

# Add time-based features to dataframe
df['time_idx'] = (df['date'] - df['date'].min()).dt.total_seconds() / 3600.0  # hours since start
df['hour_of_day'] = df['date'].dt.hour + df['date'].dt.minute/60
df['hour_of_week'] = (df['date'].dt.dayofweek * 24) + df['date'].dt.hour

# Use first 5000 rows for demo speed
df_sample = df.iloc[:5000]

# Define target and features
y = df_sample['OT']  # y = Oil Temperature column
X = df_sample[['time_idx', 'hour_of_day', 'hour_of_week']]  # X = time-based feature columns

# Fit GAM with multiple smooths
gam = LinearGAM(
    s(0, n_splines=30) +               # smooth function of time_idx
    s(1, n_splines=15, spline_order=4, basis='cp') +  # smooth function of hour_of_day (cyclic daily pattern)
    s(2, n_splines=20, spline_order=4, basis='cp')    # smooth function of hour_of_week (cyclic weekly pattern)
).fit(X, y)  # train the model: find the spline weights that best predict y

# Plot overall fitted curve
# Where gray dots are recorded data and red line is GAM fit
plt.figure(figsize=(10,5))
plt.scatter(df_sample['time_idx'], y, color='gray', alpha=0.3, label='Data')
plt.plot(df_sample['time_idx'], gam.predict(X), color='red', linewidth=2, label='GAM fit')
plt.xlabel('Time (hours since start)')
plt.ylabel('Oil Temperature (OT)')
plt.title('GAM Fit with Daily + Weekly + Long-Term Components')
plt.legend()
plt.show()

# Plot partial effects for interpretability
fig, axs = plt.subplots(3, 1, figsize=(10, 10))
titles = ['Long-term trend', 'Daily seasonality', 'Weekly seasonality']
for i, ax in enumerate(axs):
    # Make a grid of values for the i-th model term
    # Explanation: Makes a grid of evenly-spaced x-values across the range of this feature (time_idx, hour_of_day, or hour_of_week)
    XX = gam.generate_X_grid(term=i)  # Matrix dimensions = (n_grid_points, n_features)

    # Plot the smooth function's ( f_i(x_i) ) estimated contribution of the specified term for each row in X
    # partial.dependence computes the actual values of that smooth function at each x in XX
    ax.plot(XX[:, i], gam.partial_dependence(term=i, X=XX))  
    ax.set_title(titles[i])
    ax.set_xlabel(X.columns[i])
    ax.set_ylabel('Effect on OT')

plt.tight_layout()
plt.show()
