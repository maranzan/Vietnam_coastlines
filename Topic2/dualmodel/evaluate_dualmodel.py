import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. EVALUATION CONFIGURATION
# ---------------------------------------------------------
DATA_PATH = 'data/vietnam_monthly_dataset.csv'
PREDICT_MONTHS = 12       # Standard prediction horizon (1 year)
MIN_TOTAL_MONTHS = 30     # Require history + future test data

print("=" * 60)
print("  STARTING DUAL-MODEL AUTOREGRESSIVE EVALUATION")
print(f"  Prediction horizon : {PREDICT_MONTHS} months")
print("=" * 60)

# Load and prepare data
df = pd.read_csv(DATA_PATH)
df['YearMonth'] = pd.to_datetime(df['YearMonth'].astype(str))
df = df.sort_values(by=['segment_id', 'YearMonth']).reset_index(drop=True)
df['site_name'] = df['site_name'].astype('category')

# Define features, strictly excluding the target and the routing label
features = [
    'site_name', 'orientation_deg', 'slope_deg', 'elevation_m', 'dist_river_m',
    'Hs_mean_7d', 'Hs_max_7d', 'wave_period_s', 'wave_dir_deg', 
    'wind_mean_7d', 'wind_dir_deg'
]

# ---------------------------------------------------------
# 2. IDENTIFYING VALID TRANSECTS
# ---------------------------------------------------------
counts = df['segment_id'].value_counts()
valid_transects = counts[counts >= MIN_TOTAL_MONTHS].index.tolist()

if not valid_transects:
    raise ValueError(f"No transect has {MIN_TOTAL_MONTHS} months of data.")

print(f"Number of qualified transects for testing: {len(valid_transects)}\n")

# Base configurations from your training script
params_coast = {
    'n_estimators': 500, 
    'learning_rate': 0.05, 
    'max_depth': 5, 
    'enable_categorical': True, 
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

params_river = {
    'n_estimators': 400, 
    'learning_rate': 0.05, 
    'max_depth': 4, 
    'enable_categorical': True, 
    'tree_method': 'hist',
    'random_state': 42,
    'n_jobs': -1
}

# ---------------------------------------------------------
# 3. EVALUATION LOOP
# ---------------------------------------------------------
results = []
start_time = time.time()

for i, target in enumerate(valid_transects):
    transect_data = df[df['segment_id'] == target].copy()
    
    # Identify the zone to route to the correct model
    is_river = transect_data['in_river_zone'].iloc[0] == 1
    
    # Split Past / Future
    history_data = transect_data.iloc[:-PREDICT_MONTHS]
    future_data  = transect_data.iloc[-PREDICT_MONTHS:]
    
    # Prevent data leakage: Hide the future of THIS transect
    # And filter the training data to match the zone (Coast vs River)
    train_mask = (df['in_river_zone'] == (1 if is_river else 0)) & \
                 ~((df['segment_id'] == target) & (df['YearMonth'] >= future_data['YearMonth'].iloc[0]))
    
    train_df = df[train_mask]
    X_train = train_df[features]
    y_train = train_df['monthly_change_m']
    
    # Instantiate and train the routed model
    if is_river:
        model = XGBRegressor(**params_river)
    else:
        model = XGBRegressor(**params_coast)
        
    model.fit(X_train, y_train)
    
    # Autoregressive simulation
    current_position = history_data['cross_distance_m'].iloc[-1]
    predicted_positions = []
    
    for month in range(PREDICT_MONTHS):
        current_weather = future_data[features].iloc[[month]]
        predicted_change = model.predict(current_weather)[0]
        current_position += predicted_change
        predicted_positions.append(current_position)
        
    # Calculate physical errors
    true_positions = future_data['cross_distance_m'].values
    rmse = np.sqrt(np.mean((np.array(predicted_positions) - true_positions) ** 2))
    mae = np.mean(np.abs(np.array(predicted_positions) - true_positions))
    
    results.append({
        'transect': target,
        'site': target.split('_TS_')[0],
        'zone': 'River' if is_river else 'Coast',
        'rmse': rmse,
        'mae': mae
    })
    
    if (i + 1) % 50 == 0 or (i + 1) == len(valid_transects):
        print(f"Progress: {i + 1} / {len(valid_transects)} transects evaluated...")

# ---------------------------------------------------------
# 4. STATISTICS AND FINAL REPORT
# ---------------------------------------------------------
df_results = pd.DataFrame(results)
execution_time = time.time() - start_time

print("\n" + "=" * 60)
print("                 PERFORMANCE REPORT")
print("=" * 60)
print(f"Execution time         : {execution_time:.1f} seconds")
print(f"Tested transects       : {len(df_results)}")
print("-" * 60)

# Global metrics
print(f"Global RMSE (Mean)     : {df_results['rmse'].mean():.2f} meters")
print(f"Global MAE (Mean)      : {df_results['mae'].mean():.2f} meters")
print("-" * 60)

# Segmented metrics
coast_results = df_results[df_results['zone'] == 'Coast']
river_results = df_results[df_results['zone'] == 'River']

if not coast_results.empty:
    print(f"Coastal RMSE (Mean)    : {coast_results['rmse'].mean():.2f} meters ({len(coast_results)} transects)")
if not river_results.empty:
    print(f"River RMSE (Mean)      : {river_results['rmse'].mean():.2f} meters ({len(river_results)} transects)")

print("-" * 60)

# Success distribution
excellent = len(df_results[df_results['rmse'] <= 10.0])
acceptable = len(df_results[(df_results['rmse'] > 10.0) & (df_results['rmse'] <= 20.0)])
failed = len(df_results[df_results['rmse'] > 20.0])

print("Distribution of prediction scores (1-Year Horizon):")
print(f"[Excellent]  (< 10m)   : {excellent} transects ({excellent/len(df_results)*100:.1f}%)")
print(f"[Acceptable] (10-20m)  : {acceptable} transects ({acceptable/len(df_results)*100:.1f}%)")
print(f"[Failed]     (> 20m)   : {failed} transects ({failed/len(df_results)*100:.1f}%)")
print("=" * 60)