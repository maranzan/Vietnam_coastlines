import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

print("--- Fast Multivariate Preparation ---")

# 1. Load Data
df = pd.read_csv('data/vietnam_master_multivariate.csv', parse_dates=['dates'])

# 2. Reshape (Melt) - Operation vectorisée (très rapide)
id_vars = ['dates', 'site_name', 'wave_height', 'wave_period', 'wind_speed', 'tide_level']
ts_cols = [c for c in df.columns if c not in id_vars]

print("Restructuring data...")
df_long = pd.melt(df, id_vars=id_vars, value_vars=ts_cols, 
                  var_name='transect_id', value_name='distance')

# 3. Clean and Unique ID
df_long = df_long.dropna(subset=['distance', 'wave_height', 'wind_speed', 'wave_period'])
df_long['unique_id'] = df_long['site_name'] + "_" + df_long['transect_id']

# 4. OPTIMIZED CLEANING (No more slow 'for' loop)
print("Cleaning and Smoothing (Vectorized)...")

# On trie pour que le lissage temporel soit correct
df_long = df_long.sort_values(['unique_id', 'dates'])

# Au lieu d'une boucle for, on utilise groupby + transform
# Cela applique l'opération sur chaque transect de manière ultra-optimisée
target_cols = ['distance', 'wave_height', 'wind_speed', 'wave_period']

# Interpolation et Moyenne Glissante par groupe
for col in target_cols:
    # Interpolation linéaire
    df_long[col] = df_long.groupby('unique_id')[col].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both')
    )
    # Lissage (Rolling mean window=3)
    df_long[col] = df_long.groupby('unique_id')[col].transform(
        lambda x: x.rolling(window=3, min_periods=1, center=True).mean()
    )

# 5. Normalization
print("Normalizing features...")
scaler = MinMaxScaler()
df_long[target_cols] = scaler.fit_transform(df_long[target_cols])

# 6. Final Save
os.makedirs('data', exist_ok=True)
df_long.to_csv('data/dataset_ready_for_ia.csv', index=False)

with open('data/scaler_multi.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print(f"Done! Processed {len(df_long)} rows.")