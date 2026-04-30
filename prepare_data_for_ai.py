import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

# --- 1. LOAD DATA ---
try:
    df = pd.read_csv('data/vietnam_master_dataset.csv', parse_dates=['dates'])
    df = df.sort_values(['site_name', 'dates'])
    print("Master dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'data/vietnam_master_dataset.csv' not found.")
    exit()

# --- 2. DATA RESTRUCTURING (Wide to Long Format) ---
ts_cols = [c for c in df.columns if c.startswith('TS_')]

df_long = pd.melt(
    df, 
    id_vars=['dates', 'site_name'], 
    value_vars=ts_cols, 
    var_name='transect_id', 
    value_name='distance'
)

df_long = df_long.dropna(subset=['distance'])

df_long['unique_id'] = df_long['site_name'] + "_" + df_long['transect_id']

unique_transects_count = df_long['unique_id'].nunique()
print(f"Found {unique_transects_count} active transects across all sites.")

# --- 3. CLEANING & SMOOTHING BY INDIVIDUAL TRANSECT ---
processed_list = []
for tid in df_long['unique_id'].unique():
    subset = df_long[df_long['unique_id'] == tid].copy().sort_values('dates')
    
    if len(subset) < 5:
        continue
        
    subset['distance'] = subset['distance'].interpolate(method='linear', limit_direction='both')
    
    subset['distance'] = subset['distance'].rolling(window=3, min_periods=1, center=True).mean()
    
    processed_list.append(subset)

df_final = pd.concat(processed_list)

# --- 4. NORMALIZATION (MinMax Scaling) ---
scaler = MinMaxScaler(feature_range=(0, 1))

df_final['distance_scaled'] = scaler.fit_transform(df_final[['distance']])

# --- 5. SAVING OUTPUTS ---
os.makedirs('data', exist_ok=True)

df_final.to_csv('data/dataset_ready_for_ia.csv', index=False)

with open('data/scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("-" * 30)
print(f"Final Data Range: Min={df_final['distance_scaled'].min():.2f} | Max={df_final['distance_scaled'].max():.2f}")
print(f"Success! Cleaned dataset saved to 'data/dataset_ready_for_ia.csv'")
print(f"Scaler saved to 'data/scaler.pkl' (don't lose this!)")