import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

df = pd.read_csv('data/vietnam_master_dataset.csv', parse_dates=['dates'])
df = df.sort_values(['site_name', 'dates'])

ts_cols = [c for c in df.columns if c.startswith('TS_')]
valid_cols = [c for c in ts_cols if df[c].isnull().mean() < 0.5]
print(f"Keeping {len(valid_cols)} transects on {len(ts_cols)}")

processed_data = []

for site in df['site_name'].unique():
    site_df = df[df['site_name'] == site].copy()
    site_df[valid_cols] = site_df[valid_cols].interpolate(method='linear', limit_direction='both')
    site_df[valid_cols] = site_df[valid_cols].rolling(window=3, min_periods=1, center=True).mean()
    
    processed_data.append(site_df)

df_clean = pd.concat(processed_data)

scaler = MinMaxScaler()
df_clean[valid_cols] = scaler.fit_transform(df_clean[valid_cols])

pydf_clean.to_csv('dataset_ready_for_ia.csv', index=False)
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

