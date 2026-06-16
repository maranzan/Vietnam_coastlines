import pandas as pd
import numpy as np
from xgboost import XGBRegressor
import warnings
import time

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# 1. BENCHMARK CONFIGURATION
# ---------------------------------------------------------
DATA_PATH = 'data/vietnam_monthly_dataset.csv'
PREDICT_MONTHS = 12       # Standard prediction horizon (1 year)
MIN_TOTAL_MONTHS = 30     # Require at least 30 months of total data (18 history + 12 test)

print("=" * 60)
print("  STARTING XGBOOST AUTOREGRESSIVE BENCHMARK")
print(f"  Prediction horizon : {PREDICT_MONTHS} months")
print("=" * 60)

# Load and prepare data
df = pd.read_csv(DATA_PATH)
df['YearMonth'] = pd.to_datetime(df['YearMonth'].astype(str))
df = df.sort_values(by=['segment_id', 'YearMonth']).reset_index(drop=True)
df['site_name'] = df['site_name'].astype('category')

features = [
    'site_name', 'orientation_deg', 'slope_deg', 'elevation_m', 'dist_river_m',
    'Hs_mean_7d', 'Hs_max_7d', 'wave_period_s', 'wave_dir_deg', 
    'wind_mean_7d', 'wind_dir_deg'
]

# ---------------------------------------------------------
# 2. IDENTIFYING VALID TRANSECTS
# ---------------------------------------------------------
# Count the available months for each transect
counts = df['segment_id'].value_counts()
valid_transects = counts[counts >= MIN_TOTAL_MONTHS].index.tolist()

if not valid_transects:
    raise ValueError(f"No transect has {MIN_TOTAL_MONTHS} months of data. Lower the MIN_TOTAL_MONTHS filter.")

print(f"Number of qualified transects for testing: {len(valid_transects)}\n")

# ---------------------------------------------------------
# 3. EVALUATION LOOP
# ---------------------------------------------------------
results = []
start_time = time.time()

# Instantiate the base model once with the right parameters
base_model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    enable_categorical=True, 
    tree_method='hist',
    random_state=42,
    n_jobs=-1 # Use all CPU cores
)

for i, target in enumerate(valid_transects):
    transect_data = df[df['segment_id'] == target].copy()
    
    # Split Past / Future
    history_data = transect_data.iloc[:-PREDICT_MONTHS]
    future_data  = transect_data.iloc[-PREDICT_MONTHS:]
    
    # Train without data leakage (explicitly hide the future of THIS transect)
    train_df = df[~((df['segment_id'] == target) & (df['YearMonth'] >= future_data['YearMonth'].iloc[0]))]
    X_train = train_df[features]
    y_train = train_df['monthly_change_m']
    
    # Train a fresh model for this specific transect
    model = XGBRegressor(**base_model.get_params())
    model.fit(X_train, y_train)
    
    # Autoregressive simulation
    current_position = history_data['cross_distance_m'].iloc[-1]
    predicted_positions = []
    
    for month in range(PREDICT_MONTHS):
        current_weather = future_data[features].iloc[[month]]
        predicted_change = model.predict(current_weather)[0]
        current_position += predicted_change
        predicted_positions.append(current_position)
        
    # Calculate errors
    true_positions = future_data['cross_distance_m'].values
    rmse = np.sqrt(np.mean((np.array(predicted_positions) - true_positions) ** 2))
    mae = np.mean(np.abs(np.array(predicted_positions) - true_positions))
    
    results.append({
        'transect': target,
        'site': target.split('_TS_')[0],
        'rmse': rmse,
        'mae': mae
    })
    
    # Simple progress bar
    if (i + 1) % 10 == 0 or (i + 1) == len(valid_transects):
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
print(f"Global RMSE (Mean)     : {df_results['rmse'].mean():.2f} meters")
print(f"Median RMSE            : {df_results['rmse'].median():.2f} meters")
print(f"Global MAE (Mean)      : {df_results['mae'].mean():.2f} meters")
print("-" * 60)

# Success percentage (Considering an error < 10m over 1 year as very good)
excellent = len(df_results[df_results['rmse'] <= 10.0])
acceptable = len(df_results[(df_results['rmse'] > 10.0) & (df_results['rmse'] <= 20.0)])
failed = len(df_results[df_results['rmse'] > 20.0])

print("Distribution of prediction scores:")
print(f"🟢 Excellent (< 10m)   : {excellent} transects ({excellent/len(df_results)*100:.1f}%)")
print(f"🟡 Acceptable (10-20m) : {acceptable} transects ({acceptable/len(df_results)*100:.1f}%)")
print(f"🔴 Failed     (> 20m)  : {failed} transects ({failed/len(df_results)*100:.1f}%)")
print("=" * 60)

# Optional: Display the easiest/hardest sites to predict
print("\nPerformance by Site (Mean RMSE):")
print(df_results.groupby('site')['rmse'].mean().sort_values().round(2).to_string())
print("=" * 60)