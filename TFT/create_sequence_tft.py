import pandas as pd
import numpy as np
import os

# --- UPDATED CONFIGURATION ---
SEQ_LENGTH    = 90   # Aligned with the new TFT memory
PREDICT_STEPS = 36
TEST_RATIO    = 0.15 # Keep 15% "blind" transects for the final test

print("Loading dataset...")
df = pd.read_csv('data/vietnam_master_multivariate.csv')

# Reshape wide → long
print("Reshaping data...")
id_vars = ['dates', 'site_name', 'wave_height', 'wave_period', 'wind_speed', 'tide_level']
transect_cols = [c for c in df.columns if c not in id_vars]
df = df.melt(id_vars=id_vars, value_vars=transect_cols,
             var_name='transect_id', value_name='distance')
df['unique_id'] = df['site_name'] + "_" + df['transect_id']

# Clean missing values on key features
feature_cols = ['distance', 'wave_height', 'wind_speed', 'wave_period', 'tide_level']
df = df.dropna(subset=feature_cols)
df['dates'] = pd.to_datetime(df['dates'])
df = df.sort_values(['unique_id', 'dates']).reset_index(drop=True)

# --- ROBUST OUTLIER REMOVAL (CORRECTED) ---
print("Removing outliers...")
def remove_outliers(group):
    group = group.copy()
    
    # 1. Global Filter: Instantly drop wild CoastSat errors (> 150m from global median)
    global_median = group['distance'].median()
    is_extreme = (group['distance'] - global_median).abs() > 150
    group.loc[is_extreme, 'distance'] = np.nan
    
    # 2. Local Filter (Rolling median)
    # window=5 allows smoothing over roughly 5 satellite images
    rolling_median = group['distance'].rolling(window=5, center=True, min_periods=1).median()
    
    # Physical threshold: If a point jumps more than 35 meters suddenly compared to local trend, it's likely a cloud.
    # Adjust (35-50m) depending on how dynamic your beaches are.
    is_local_outlier = (group['distance'] - rolling_median).abs() > 35
    group.loc[is_local_outlier, 'distance'] = np.nan
    
    # 3. Interpolation to fill the gaps (removed clouds)
    group['distance'] = group['distance'].interpolate(method='linear')
    
    # If the very first or last point was NaN, linear interpolation misses it, so we use bfill/ffill
    group['distance'] = group['distance'].bfill().ffill()
    
    return group

# APPLY THE OUTLIER REMOVAL (Added this line back in so the function is actually used)
df = df.groupby('unique_id', group_keys=False).apply(remove_outliers)

# --- TEMPORAL FEATURES ---
df['month_sin'] = np.sin(2 * np.pi * df['dates'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['dates'].dt.month / 12)

# --- STRICT REAL TIME INDEX ---
print("Calculating strict time index...")
# Based on the global minimum date so index 1 is the exact same day everywhere
min_global_date = df['dates'].min()
df['time_idx'] = (df['dates'] - min_global_date).dt.days

# --- FILTER SHORT TRANSECTS ---
# A transect must have enough points to cover history + prediction
min_length = SEQ_LENGTH + PREDICT_STEPS + 10
valid_ids = df.groupby('unique_id').filter(
    lambda g: len(g) >= min_length
)['unique_id'].unique()
df = df[df['unique_id'].isin(valid_ids)].reset_index(drop=True)

# --- SPATIAL SPLIT ONLY FOR INDEPENDENT TEST ---
np.random.seed(42)
all_ids = np.random.permutation(valid_ids)
n = len(all_ids)

# Isolate 15% of transects that will NEVER be seen by the model (Pure Test)
n_test = int(TEST_RATIO * n)
test_ids = all_ids[:n_test]
train_val_ids = all_ids[n_test:] # These transects will be used for temporal train/val

# Final dataset filtering: keep everything in CSV, train_tft will handle the rest
print(f"Total transects : {n}")
print(f"  Train/Val (Temporal split later) : {len(train_val_ids)} transects")
print(f"  Pure Test (Spatially unseen)     : {len(test_ids)} transects")

# --- SAVE ---
os.makedirs('data', exist_ok=True)

# Save the complete dataset
df.to_csv('data/dataset_tft.csv', index=False)

# Save the list of test IDs for the inference script
np.save('data/test_ids.npy', test_ids)

# Save train_val_ids just in case, even though train will use all by default
np.save('data/train_val_ids.npy', train_val_ids)

print("=" * 50)
print(f"Total recorded observations : {len(df):,}")
print("Saved → data/dataset_tft.csv + test_ids.npy")
print("=" * 50)