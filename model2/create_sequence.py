import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# 1. Load the dataset
print("Loading dataset...")
df = pd.read_csv('data/vietnam_master_multivariate.csv')

# 2. Reshape from wide to long format (Melting)
print("Reshaping data (Wide to Long)...")
id_vars = ['dates', 'site_name', 'wave_height', 'wave_period', 'wind_speed', 'tide_level']
transect_cols = [c for c in df.columns if c not in id_vars]
df = df.melt(id_vars=id_vars, value_vars=transect_cols,
             var_name='transect_id', value_name='distance')

df['unique_id'] = df['site_name'] + "_" + df['transect_id']

# 3. Clean NaNs
print("Global NaN scrubbing...")
feature_cols = ['distance', 'wave_height', 'wind_speed', 'wave_period']
df = df.dropna(subset=feature_cols)

# 4. Normalization des features (MinMaxScaler comme avant)
print("Normalizing features...")
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

os.makedirs('data', exist_ok=True)
with open('data/scaler_delta.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("Feature scaler saved → data/scaler_delta.pkl")

df = df.sort_values(['unique_id', 'dates'])

# 5. VECTORIZED SEQUENCING AVEC TARGET DELTA
print("Generating sequences with DELTA target...")
seq_length = 10

features_array = df[feature_cols].values.astype(np.float32)
distance_array = df['distance'].values.astype(np.float32)
ids_array = df['unique_id'].values

# Calcul des deltas bruts (dans l'espace scalé MinMax)
delta_array = np.diff(distance_array, prepend=distance_array[0]).astype(np.float32)

# Sliding windows sur les features
X_windows = np.lib.stride_tricks.sliding_window_view(
    features_array, window_shape=(seq_length, len(feature_cols))
).squeeze()

# Target = delta juste après la fenêtre
y_windows = delta_array[seq_length:]
X_windows = X_windows[:-1]

# Filtrage des frontières entre transects
id_encoded = pd.factorize(ids_array)[0]
id_changes = np.diff(id_encoded) != 0
change_windows = np.lib.stride_tricks.sliding_window_view(id_changes, window_shape=seq_length)
valid_sequence_mask = ~change_windows.any(axis=1)

# Filtrage des deltas aberrants (> 3 std)
delta_std = delta_array.std()
delta_mean = delta_array.mean()
valid_delta_mask = np.abs(y_windows) < (delta_mean + 3 * delta_std)

final_mask = valid_sequence_mask & valid_delta_mask

X = X_windows[final_mask]
y = y_windows[final_mask].reshape(-1, 1)

# --- NORMALISATION DES DELTAS avec StandardScaler ---
# StandardScaler centre sur 0 → parfait pour des deltas qui oscillent autour de 0
print("Normalizing delta targets with StandardScaler...")
delta_scaler = StandardScaler()
y = delta_scaler.fit_transform(y)

with open('data/delta_scaler.pkl', 'wb') as f:
    pickle.dump(delta_scaler, f)
print("Delta scaler saved → data/delta_scaler.pkl")

# 6. Export
print("Saving clean matrices...")
np.save('data/X_train_delta.npy', X)
np.save('data/y_train_delta.npy', y)

print("=" * 50)
print(f"Success! X shape : {X.shape}")
print(f"         y shape : {y.shape}")
print(f"Delta mean (normalized) : {y.mean():.6f}  ← doit être ~0")
print(f"Delta std  (normalized) : {y.std():.6f}   ← doit être ~1")
print("=" * 50)