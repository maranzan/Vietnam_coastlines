import pandas as pd
import numpy as np
import os
import json
import warnings
import random
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from xgboost import XGBRegressor
import geopandas as gpd
from shapely.geometry import Point

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = 'data/vietnam_ml_dataset.csv'
MODEL_DIR = 'models'
FORECAST_MONTHS = 24  
NUM_PLOTS = 3         

print("1. Loading Data and Pre-Trained Models...")
# Load feature order
with open(f'{MODEL_DIR}/training_features.json', 'r') as f:
    training_features = json.load(f)

# Load XGBoost models
model_coast = XGBRegressor()
model_coast.load_model(f'{MODEL_DIR}/coast_model.json')

model_river = XGBRegressor()
model_river.load_model(f'{MODEL_DIR}/river_model.json')

# Load historical data to establish baseline and climatology
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df = df.sort_values(by=['segment_id', 'date'])

df['lag_1_change'] = df.groupby('segment_id')['shoreline_change_m'].shift(1)
df['rolling_3_change'] = df.groupby('segment_id')['shoreline_change_m'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)
df['site_name'] = df['site_name'].astype('category')
df['satellite'] = df['satellite'].astype('category') # Fix pour XGBoost categorical

# --- CLIMATOLOGY (Simulate future weather) ---
print("2. Calculating seasonal climatology...")
weather_cols = ['Hs_mean_7d', 'Hs_max_7d', 'wave_period_s', 'wave_dir_deg', 'wind_mean_7d', 'wind_dir_deg']
climatology = df.groupby(['segment_id', 'month'])[weather_cols].mean().reset_index()

# Select high-quality transects for visualization
counts = df['segment_id'].value_counts()
valid_transects = counts[counts >= 30].index.tolist()
TRANSECTS_TO_PLOT = random.sample(valid_transects, min(NUM_PLOTS, len(valid_transects)))
forecast_trajectories = {seg: {'dates': [], 'positions': []} for seg in TRANSECTS_TO_PLOT}

# --- PREPARE AUTOREGRESSIVE ENGINE ---
print(f"3. Launching simulation for {FORECAST_MONTHS} months...")
last_known_state = df.drop_duplicates(subset=['segment_id'], keep='last').copy()

current_positions = last_known_state.set_index('segment_id')['cross_distance_m'].to_dict()
recent_changes = {seg: [row['rolling_3_change'], row['lag_1_change']] 
                  for seg, row in last_known_state.set_index('segment_id').iterrows()}
total_forecasted_change = {seg: 0.0 for seg in last_known_state['segment_id']}

current_date = last_known_state['date'].max()

for step in range(1, FORECAST_MONTHS + 1):
    current_date += pd.DateOffset(months=1)
    current_month = current_date.month
    
    step_df = last_known_state.copy()
    step_df['date'] = current_date
    step_df['month'] = current_month
    
    # Inject standard weather
    step_df = step_df.drop(columns=weather_cols)
    step_df = step_df.merge(climatology[climatology['month'] == current_month], on=['segment_id', 'month'], how='left')
    
    # Update autoregressive memory
    step_df['lag_1_change'] = step_df['segment_id'].map(lambda x: recent_changes[x][-1])
    step_df['rolling_3_change'] = step_df['segment_id'].map(lambda x: np.mean(recent_changes[x][-3:]))
    
    mask_coast = step_df['in_river_zone'] == 0
    mask_river = step_df['in_river_zone'] == 1
    
    # Batch Prediction
    if mask_coast.sum() > 0:
        X_c = step_df.loc[mask_coast, training_features]
        step_df.loc[mask_coast, 'pred_change'] = model_coast.predict(X_c)
        
    if mask_river.sum() > 0:
        X_r = step_df.loc[mask_river, training_features]
        step_df.loc[mask_river, 'pred_change'] = model_river.predict(X_r)
        
    # State update
    for _, row in step_df.iterrows():
        seg = row['segment_id']
        change = row['pred_change']
        
        recent_changes[seg].append(change)
        if len(recent_changes[seg]) > 3:
            recent_changes[seg].pop(0)
            
        current_positions[seg] += change
        total_forecasted_change[seg] += change
        
        if seg in TRANSECTS_TO_PLOT:
            forecast_trajectories[seg]['dates'].append(current_date)
            forecast_trajectories[seg]['positions'].append(current_positions[seg])

    if step % 6 == 0:
        print(f"   Progress: {step}/{FORECAST_MONTHS} months projected...")

# --- GIS EXPORT PIPELINE ---
print("\n4. GIS Export Pipeline (GeoJSON & Shapefile)...")
gis_df = last_known_state.copy()
gis_df['total_predicted_change_m'] = gis_df['segment_id'].map(total_forecasted_change)
gis_df['final_position_m'] = gis_df['segment_id'].map(current_positions)
gis_df['model_used'] = np.where(gis_df['in_river_zone'] == 1, 'River/Estuary', 'Coastal')

# Clean types for Shapefile compatibility
gis_df['date'] = gis_df['date'].astype(str)
# Ligne gis_df['YearMonth'] supprimée car obsolète

geometry = [Point(xy) for xy in zip(gis_df['longitude'], gis_df['latitude'])]
gdf = gpd.GeoDataFrame(gis_df, geometry=geometry, crs="EPSG:4326")

cols_to_keep = [
    'segment_id', 'site_name', 'latitude', 'longitude', 'model_used', 
    'total_predicted_change_m', 'final_position_m', 'geometry'
]
gdf_clean = gdf[cols_to_keep]

os.makedirs('data/gis_exports', exist_ok=True)
gdf_clean.to_file("data/gis_exports/vietnam_forecast_24m.geojson", driver="GeoJSON")
print("-> GeoJSON exported successfully.")
gdf_clean.to_file("data/gis_exports/vietnam_forecast_24m.shp", driver="ESRI Shapefile")
print("-> Shapefile exported successfully.")

# --- GENERATE GRAPHS ---
print("\n5. Generating prospective graphs...")
for seg in TRANSECTS_TO_PLOT:
    history_data = df[df['segment_id'] == seg].sort_values('date')
    plt.figure(figsize=(12, 6))
    
    plot_history = history_data.tail(36)
    plt.plot(plot_history['date'], plot_history['cross_distance_m'], label='Known History', color='black', marker='o', markersize=4)
    
    pred_dates = [plot_history['date'].iloc[-1]] + forecast_trajectories[seg]['dates']
    pred_vals = [plot_history['cross_distance_m'].iloc[-1]] + forecast_trajectories[seg]['positions']
    
    plt.plot(pred_dates, pred_vals, label=f'AI Projection ({FORECAST_MONTHS} months)', color='red', linestyle='--', marker='x', markersize=6)
    
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

print("\nForecasting Phase Completed!")