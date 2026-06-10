import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. SIMULATION CONFIGURATION
# ---------------------------------------------------------
DATA_PATH = 'data/vietnam_monthly_dataset.csv'
TARGET_TRANSECT = 'Mui_Ne_TS_0006'  # Change
DESIRED_PREDICT_MONTHS = 12

print(f"Loading data for transect: {TARGET_TRANSECT}...")
df = pd.read_csv(DATA_PATH)

df['YearMonth'] = pd.to_datetime(df['YearMonth'].astype(str))
df = df.sort_values(by=['segment_id', 'YearMonth']).reset_index(drop=True)

# Convert strings to categorical types for XGBoost
df['site_name'] = df['site_name'].astype('category')

# Features required by the model
features = [
    'site_name', 'orientation_deg', 'slope_deg', 'elevation_m', 'dist_river_m',
    'Hs_mean_7d', 'Hs_max_7d', 'wave_period_s', 'wave_dir_deg', 
    'wind_mean_7d', 'wind_dir_deg'
]

# ---------------------------------------------------------
# 2. DATA SPLIT (History vs Future) WITH DYNAMIC SIZING
# ---------------------------------------------------------

transect_data = df[df['segment_id'] == TARGET_TRANSECT].copy()
total_months = len(transect_data)

if total_months < 12:
    raise ValueError(f"Critical lack of data: Only {total_months} valid months found for {TARGET_TRANSECT}. Need at least 12 months.")

# Dynamically adjust the prediction window if we don't have enough data for 24 months
# We keep roughly 2/3 of the data as history, and predict the remaining 1/3
actual_predict_months = min(DESIRED_PREDICT_MONTHS, total_months // 3)

print(f"Total valid months available for this transect: {total_months}")
print(f"Simulating the last {actual_predict_months} months...")

# Split into Past (History) and Future (To predict)
history_data = transect_data.iloc[:-actual_predict_months]
future_data  = transect_data.iloc[-actual_predict_months:]


train_df = df[~((df['segment_id'] == TARGET_TRANSECT) & (df['YearMonth'] >= future_data['YearMonth'].iloc[0]))]

X_train = train_df[features]
y_train = train_df['monthly_change_m']

# ---------------------------------------------------------
# 3. MODEL TRAINING
# ---------------------------------------------------------
print("Training the XGBoost model on historical data...")
model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    enable_categorical=True, 
    tree_method='hist',
    random_state=42
)
model.fit(X_train, y_train)

# ---------------------------------------------------------
# 4. AUTOREGRESSIVE SIMULATION
# ---------------------------------------------------------
print("Running the step-by-step forecasting loop...")

# Starting point: The last known absolute position of the beach
current_position = history_data['cross_distance_m'].iloc[-1]

predicted_positions = []
true_positions = future_data['cross_distance_m'].values
future_dates = future_data['YearMonth'].values
history_dates = history_data['YearMonth'].values
history_positions = history_data['cross_distance_m'].values

# The loop: Advancing one month at a time
for i in range(actual_predict_months):
    current_weather = future_data[features].iloc[[i]]
    predicted_change = model.predict(current_weather)[0]
    current_position += predicted_change
    predicted_positions.append(current_position)

# ---------------------------------------------------------
# 5. RESULTS & VISUALIZATION
# ---------------------------------------------------------
rmse = np.sqrt(np.mean((np.array(predicted_positions) - true_positions) ** 2))
mae = np.mean(np.abs(np.array(predicted_positions) - true_positions))

print("\n" + "=" * 55)
print(f"XGBOOST SIMULATION RESULTS : {TARGET_TRANSECT}")
print("=" * 55)
print(f"Trajectory RMSE: {rmse:.2f} m  |  MAE: {mae:.2f} m")
print("=" * 55)

# Plotting
plt.figure(figsize=(14, 7))

# Limit history visualization to keep the plot readable
plot_hist_len = min(36, len(history_dates))

# Plot historical truth
plt.plot(history_dates[-plot_hist_len:], history_positions[-plot_hist_len:], 
         label='History (CoastSat)', color='black', marker='o', linewidth=2)

# Plot future truth
plt.plot(future_dates, true_positions, 
         label='True Future (CoastSat)', color='purple', alpha=0.9, linewidth=2.5, marker='s')

# Plot AI Prediction
plt.plot(future_dates, predicted_positions, 
         label='XGBoost Prediction (Real Weather Force)', color='blue', linestyle='--', linewidth=2)

# Connect the lines visually from the last known point
plt.plot([history_dates[-1], future_dates[0]], [history_positions[-1], predicted_positions[0]], 
         color='blue', linestyle='--', linewidth=2)
plt.plot([history_dates[-1], future_dates[0]], [history_positions[-1], true_positions[0]], 
         color='purple', alpha=0.9, linewidth=2.5)

plt.title(f'Autoregressive Simulation ({actual_predict_months} months) — {TARGET_TRANSECT}')
plt.xlabel('Date')
plt.ylabel('Shoreline Position (meters)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()