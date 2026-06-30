import pandas as pd
import numpy as np
import os
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = 'data/vietnam_monthly_dataset.csv'
FORECAST_MONTHS = 120  # 2-year prediction into the future
NUM_PLOTS = 3         # Number of graphs to display

print("1. Loading and preparing data...")
df = pd.read_csv(DATA_PATH)

# Feature Engineering
df['date'] = pd.to_datetime(df['YearMonth'])
df['month'] = df['date'].dt.month
df = df.sort_values(by=['segment_id', 'date'])

df['lag_1_change'] = df.groupby('segment_id')['monthly_change_m'].shift(1)
df['rolling_3_change'] = df.groupby('segment_id')['monthly_change_m'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Type safety
target = 'monthly_change_m'
df['site_name'] = df['site_name'].astype('category')
cols_to_drop = ['segment_id', 'YearMonth', 'date', 'cross_distance_m', 'in_river_zone', target]

# Clean NaN values for training
df_train = df.dropna(subset=[target, 'lag_1_change', 'rolling_3_change'])

df_coast = df_train[df_train['in_river_zone'] == 0].copy()
df_river = df_train[df_train['in_river_zone'] == 1].copy()

# Save the exact feature order for inference to prevent mismatch errors
training_features = df_coast.drop(columns=cols_to_drop).columns.tolist()

# --- MODEL TRAINING ---
print("\n2. Training Coast and River models...")
params_coast = {
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 
    'subsample': 0.8, 'colsample_bytree': 0.8, 'enable_categorical': True, 
    'tree_method': 'hist', 'random_state': 42
}
model_coast = XGBRegressor(**params_coast)
model_coast.fit(df_coast.drop(columns=cols_to_drop), df_coast[target])

params_river = {
    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 4, 
    'subsample': 0.8, 'colsample_bytree': 0.8, 'enable_categorical': True, 
    'tree_method': 'hist', 'random_state': 42
}
model_river = XGBRegressor(**params_river)
model_river.fit(df_river.drop(columns=cols_to_drop), df_river[target])

# --- CLIMATOLOGY (Simulate future weather) ---
print("3. Calculating seasonal climatology (Standard future weather)...")
weather_cols = ['Hs_mean_7d', 'Hs_max_7d', 'wave_period_s', 'wave_dir_deg', 'wind_mean_7d', 'wind_dir_deg']
climatology = df.groupby(['segment_id', 'month'])[weather_cols].mean().reset_index()

# --- SELECT TRANSECTS TO VISUALIZE ---
# Filter for transects that have at least 30 months of historical data
counts = df['segment_id'].value_counts()
valid_transects = counts[counts >= 30].index.tolist()

# Select random high-quality transects to plot
TRANSECTS_TO_PLOT = random.sample(valid_transects, min(NUM_PLOTS, len(valid_transects)))
forecast_trajectories = {seg: {'dates': [], 'positions': []} for seg in TRANSECTS_TO_PLOT}

# --- PREPARE AUTOREGRESSIVE ENGINE ---
print(f"4. Launching simulation for {FORECAST_MONTHS} months...")
last_known_state = df.drop_duplicates(subset=['segment_id'], keep='last').copy()

current_positions = last_known_state.set_index('segment_id')['cross_distance_m'].to_dict()
recent_changes = {seg: [row['rolling_3_change'], row['lag_1_change']] 
                  for seg, row in last_known_state.set_index('segment_id').iterrows()}

current_date = last_known_state['date'].max()

for step in range(1, FORECAST_MONTHS + 1):
    current_date += pd.DateOffset(months=1)
    current_month = current_date.month
    
    step_df = last_known_state.copy()
    step_df['date'] = current_date
    step_df['month'] = current_month
    
    # Inject standard weather for this month
    step_df = step_df.drop(columns=weather_cols)
    step_df = step_df.merge(climatology[climatology['month'] == current_month], on=['segment_id', 'month'], how='left')
    
    # Update autoregressive variables
    step_df['lag_1_change'] = step_df['segment_id'].map(lambda x: recent_changes[x][-1])
    step_df['rolling_3_change'] = step_df['segment_id'].map(lambda x: np.mean(recent_changes[x][-3:]))
    
    mask_coast = step_df['in_river_zone'] == 0
    mask_river = step_df['in_river_zone'] == 1
    
    # Predict using the exact feature order saved earlier
    if mask_coast.sum() > 0:
        X_c = step_df.loc[mask_coast, training_features]
        step_df.loc[mask_coast, 'pred_change'] = model_coast.predict(X_c)
        
    if mask_river.sum() > 0:
        X_r = step_df.loc[mask_river, training_features]
        step_df.loc[mask_river, 'pred_change'] = model_river.predict(X_r)
        
    for _, row in step_df.iterrows():
        seg = row['segment_id']
        change = row['pred_change']
        
        # Update sliding history
        recent_changes[seg].append(change)
        if len(recent_changes[seg]) > 3:
            recent_changes[seg].pop(0)
            
        current_positions[seg] += change
        
        # Save the trajectory for the graphs
        if seg in TRANSECTS_TO_PLOT:
            forecast_trajectories[seg]['dates'].append(current_date)
            forecast_trajectories[seg]['positions'].append(current_positions[seg])

    if step % 6 == 0:
        print(f"   Progress: {step}/{FORECAST_MONTHS} months projected...")

# --- GENERATE GRAPHS ---
print("\n5. Generating graphs...")
for seg in TRANSECTS_TO_PLOT:
    history_data = df[df['segment_id'] == seg].sort_values('date')
    
    plt.figure(figsize=(12, 6))
    
    # Plot known history (last 3 years max for readability)
    plot_history = history_data.tail(36)
    plt.plot(plot_history['date'], plot_history['cross_distance_m'], 
             label='Known History', color='black', marker='o', markersize=4)
    
    # Plot prediction
    # Add the last known point to the beginning so the line is continuous
    pred_dates = [plot_history['date'].iloc[-1]] + forecast_trajectories[seg]['dates']
    pred_vals = [plot_history['cross_distance_m'].iloc[-1]] + forecast_trajectories[seg]['positions']
    
    plt.plot(pred_dates, pred_vals, 
             label=f'AI Projection ({FORECAST_MONTHS} months)', color='red', linestyle='--', marker='x', markersize=6)
    
    # Formatting
    zone_label = "River/Estuary" if history_data['in_river_zone'].iloc[0] == 1 else "Coast (Beach)"
    plt.title(f"Prospective Forecast - Transect: {seg} ({zone_label})")
    plt.xlabel("Date")
    plt.ylabel("Cross Distance to Shoreline (m)")
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=4))
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

print("\nDone!")