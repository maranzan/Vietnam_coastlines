import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import warnings
import random

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------
DATA_PATH = 'data/vietnam_monthly_dataset.csv'
PREDICT_MONTHS = 12
MIN_TOTAL_MONTHS = 30
NUM_PLOTS = 3  # How many random transects to visualize

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)
df['YearMonth'] = pd.to_datetime(df['YearMonth'].astype(str))
df = df.sort_values(by=['segment_id', 'YearMonth']).reset_index(drop=True)
df['site_name'] = df['site_name'].astype('category')

features = [
    'site_name', 'orientation_deg', 'slope_deg', 'elevation_m', 'dist_river_m',
    'Hs_mean_7d', 'Hs_max_7d', 'wave_period_s', 'wave_dir_deg', 
    'wind_mean_7d', 'wind_dir_deg'
]

# Identify valid transects
counts = df['segment_id'].value_counts()
valid_transects = counts[counts >= MIN_TOTAL_MONTHS].index.tolist()

# Select random transects to plot (or replace this list with specific IDs)
sample_transects = random.sample(valid_transects, NUM_PLOTS)

params_coast = {
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 
    'enable_categorical': True, 'tree_method': 'hist', 'random_state': 42
}

params_river = {
    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 4, 
    'enable_categorical': True, 'tree_method': 'hist', 'random_state': 42
}

# ---------------------------------------------------------
# 2. VISUALIZATION LOOP
# ---------------------------------------------------------
for target in sample_transects:
    print(f"\nProcessing Transect: {target}")
    transect_data = df[df['segment_id'] == target].copy()
    
    is_river = transect_data['in_river_zone'].iloc[0] == 1
    zone_label = "River/Estuary" if is_river else "Coastal"
    
    history_data = transect_data.iloc[:-PREDICT_MONTHS]
    future_data  = transect_data.iloc[-PREDICT_MONTHS:]
    
    # Train model
    train_mask = (df['in_river_zone'] == (1 if is_river else 0)) & \
                 ~((df['segment_id'] == target) & (df['YearMonth'] >= future_data['YearMonth'].iloc[0]))
    
    train_df = df[train_mask]
    X_train = train_df[features]
    y_train = train_df['monthly_change_m']
    
    model = XGBRegressor(**params_river) if is_river else XGBRegressor(**params_coast)
    model.fit(X_train, y_train)
    
    # Predict
    current_position = history_data['cross_distance_m'].iloc[-1]
    predicted_positions = []
    
    for month in range(PREDICT_MONTHS):
        current_weather = future_data[features].iloc[[month]]
        predicted_change = model.predict(current_weather)[0]
        current_position += predicted_change
        predicted_positions.append(current_position)
        
    true_positions = future_data['cross_distance_m'].values
    rmse = np.sqrt(np.mean((np.array(predicted_positions) - true_positions) ** 2))
    
    # ---------------------------------------------------------
    # 3. PLOTTING
    # ---------------------------------------------------------
    plt.figure(figsize=(12, 6))
    
    # Plot history (show up to 24 months of history for context)
    plot_history = history_data.tail(24)
    plt.plot(plot_history['YearMonth'], plot_history['cross_distance_m'], 
             label='Historical True Position', color='black', marker='o', markersize=4)
    
    # Plot true future
    plt.plot(future_data['YearMonth'], future_data['cross_distance_m'], 
             label='True Future Position', color='blue', marker='o', markersize=4)
    
    # Plot predicted future
    # Prepend the last historical point so the line connects seamlessly
    pred_dates = pd.concat([plot_history['YearMonth'].tail(1), future_data['YearMonth']])
    pred_vals = [plot_history['cross_distance_m'].iloc[-1]] + predicted_positions
    
    plt.plot(pred_dates, pred_vals, 
             label='Predicted Future Position', color='red', linestyle='--', marker='x', markersize=6)
    
    # Formatting
    plt.title(f"Transect: {target} ({zone_label}) | 1-Year Forecast RMSE: {rmse:.2f}m")
    plt.xlabel("Date")
    plt.ylabel("Cross Distance (meters)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    # Improve date formatting on X-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

print("\nVisualization complete.")