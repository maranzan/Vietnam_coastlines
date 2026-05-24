import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler

# --- CONFIGURATION ---
SEQ_LENGTH    = 10   # Encoder window (history)
PREDICT_STEPS = 36   # Decoder window (forecast horizon)
TEST_RATIO    = 0.15

print("Loading dataset...")
df = pd.read_csv('data/vietnam_master_multivariate.csv')

# Reshape wide → long
print("Reshaping data...")
id_vars = ['dates', 'site_name', 'wave_height', 'wave_period', 'wind_speed', 'tide_level']
transect_cols = [c for c in df.columns if c not in id_vars]
df = df.melt(id_vars=id_vars, value_vars=transect_cols,
             var_name='transect_id', value_name='distance')
df['unique_id'] = df['site_name'] + "_" + df['transect_id']
df['dates'] = pd.to_datetime(df['dates'])

# Clean NaN values on critical variables
feature_cols_base = ['distance', 'wave_height', 'wind_speed', 'wave_period', 'tide_level']
df = df.dropna(subset=feature_cols_base)

# --- ROBUST OUTLIER REMOVAL ---
print("Removing outliers...")
def remove_outliers(group):
    group = group.copy()
    
    # 1. Global Filter: Instantly drop extreme artifacts (> 150m from global median)
    global_median = group['distance'].median()
    is_extreme    = (group['distance'] - global_median).abs() > 150
    group.loc[is_extreme, 'distance'] = np.nan
    
    # 2. Local Filter: Rolling median vs threshold
    rolling_median = group['distance'].rolling(window=5, center=True, min_periods=1).median()
    is_local       = (group['distance'] - rolling_median).abs() > 35
    group.loc[is_local, 'distance'] = np.nan
    
    # 3. Interpolation and fallback padding
    group['distance'] = group['distance'].interpolate(method='linear').bfill().ffill()
    return group

df = df.sort_values(['unique_id', 'dates'])
df = df.groupby('unique_id', group_keys=False).apply(remove_outliers)
df = df.dropna(subset=feature_cols_base)

# --- TEMPORAL FEATURES ---
df['month_sin'] = np.sin(2 * np.pi * df['dates'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['dates'].dt.month / 12)

# --- NORMALIZATION ---
feature_cols_scaled = ['distance', 'wave_height', 'wind_speed', 'wave_period', 'tide_level']
feature_cols_all    = feature_cols_scaled + ['month_sin', 'month_cos']  # 7 features in total

print("Normalizing...")
scaler = MinMaxScaler()
df[feature_cols_scaled] = scaler.fit_transform(df[feature_cols_scaled])

os.makedirs('data', exist_ok=True)
with open('data/scaler_seq2seq.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Scaler saved → data/scaler_seq2seq.pkl")

df = df.sort_values(['unique_id', 'dates']).reset_index(drop=True)

# --- TRAIN / TEST SPLIT BY ENTIRE TRANSECTS ---
np.random.seed(42)
all_ids  = df['unique_id'].unique()
all_ids  = np.random.permutation(all_ids)
n        = len(all_ids)
n_test   = int(TEST_RATIO * n)

test_ids  = all_ids[:n_test]
train_ids = all_ids[n_test:]

np.save('data/test_ids_seq2seq.npy',  test_ids)
np.save('data/train_ids_seq2seq.npy', train_ids)

print(f"Total transects : {n}")
print(f"  Train : {len(train_ids)} transects")
print(f"  Test  : {len(test_ids)} transects (never seen)")

# --- SEQ2SEQ SEQUENCE GENERATION (OPTIMIZED USING SLIDING WINDOWS) ---
print("Generating Seq2Seq sequences (7 features)...")

features_array = df[feature_cols_all].values.astype(np.float32)  # (T, 7)
ids_array      = df['unique_id'].values
total_window   = SEQ_LENGTH + PREDICT_STEPS

print("Generating Seq2Seq sequences (chunked to save RAM)...")

# High-efficiency memory window view
X_windows = np.lib.stride_tricks.sliding_window_view(
    features_array, window_shape=(total_window, features_array.shape[1])
).squeeze(1)  # (N, total_window, 7)

# Boundary detection to prevent overlapping different transects
id_encoded = pd.factorize(ids_array)[0]
id_changes = np.diff(id_encoded) != 0
change_windows = np.lib.stride_tricks.sliding_window_view(
    id_changes, window_shape=total_window - 1
)
valid_mask = ~change_windows.any(axis=1)
valid_indices = np.where(valid_mask)[0]
num_valid = len(valid_indices)

print(f"Total valid sequences to extract: {num_valid:,}")

# Pre-allocating destination arrays for strict memory containment
X_enc = np.empty((num_valid, SEQ_LENGTH, 7), dtype=np.float32)
X_dec = np.empty((num_valid, PREDICT_STEPS, 7), dtype=np.float32)
y     = np.empty((num_valid, PREDICT_STEPS, 1), dtype=np.float32)

# Chunk processing loop
chunk_size = 100000
for i in range(0, num_valid, chunk_size):
    idx_chunk = valid_indices[i : i+chunk_size]
    
    # Extract blocks
    chunk_enc = X_windows[idx_chunk, :SEQ_LENGTH, :]
    chunk_dec = X_windows[idx_chunk, SEQ_LENGTH:, :]
    
    X_enc[i : i+chunk_size] = chunk_enc
    y[i : i+chunk_size]     = chunk_dec[:, :, 0:1]
    
    # Teacher forcing processing setup: shifts targets forward by 1 step
    last_known = np.empty((len(idx_chunk), PREDICT_STEPS, 1), dtype=np.float32)
    last_known[:, 0:1, :] = chunk_enc[:, -1:, 0:1]
    last_known[:, 1:, :]  = chunk_dec[:, :-1, 0:1]
    
    X_dec[i : i+chunk_size, :, 0:1] = last_known
    X_dec[i : i+chunk_size, :, 1:]  = chunk_dec[:, :, 1:]
    
    print(f"  -> Processed {min(i+chunk_size, num_valid):,} / {num_valid:,}")

seq_ids = ids_array[:len(valid_mask)][valid_mask]

# --- SEPARATING AND SAVING DISK ARRAYS ---
print("Saving files to disk... (This might take a few seconds)")
train_mask = np.isin(seq_ids, train_ids)
test_mask  = np.isin(seq_ids, test_ids)

# Save Train Data
np.save('data/X_enc_train.npy', X_enc[train_mask])
np.save('data/X_dec_train.npy', X_dec[train_mask])
np.save('data/y_train.npy',     y[train_mask])

# Save Test Data
np.save('data/X_enc_test.npy',  X_enc[test_mask])
np.save('data/X_dec_test.npy',  X_dec[test_mask])
np.save('data/y_test.npy',      y[test_mask])

print("=" * 60)
print(f"X_enc shape total : {X_enc.shape}")
print(f"Train sequences   : {train_mask.sum():,}")
print(f"Test  sequences   : {test_mask.sum():,}")
print("Saved → data/X_enc_train.npy + X_enc_test.npy (and others)")
print("=" * 60)