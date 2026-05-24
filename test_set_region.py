import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
INPUT_FILE = 'data/vietnam_master_dataset.csv'
SCALER_FILE = 'data/scaler.pkl'
OUTPUT_TEST_FILE = 'data/test_set_region.csv'
TARGET_REGION = 'QN_Tam_Thanh'  # Tu peux changer pour Nui Thanh, etc.

def prepare_test_region(region_name):
    print(f"--- Processing Region: {region_name} ---")
    
    # 1. Chargement du Master Dataset
    if not os.path.exists(INPUT_FILE):
        print(f"Error: {INPUT_FILE} not found!")
        return
    
    df = pd.read_csv(INPUT_FILE, parse_dates=['dates'])
    
    # 2. Filtrage par région
    df_region = df[df['site_name'] == region_name].copy()
    if df_region.empty:
        print(f"No data found for region: {region_name}")
        return

    # 3. Restructuration (Wide to Long)
    ts_cols = [c for c in df_region.columns if c.startswith('TS_')]
    df_long = pd.melt(
        df_region, 
        id_vars=['dates', 'site_name'], 
        value_vars=ts_cols, 
        var_name='transect_id', 
        value_name='distance'
    )
    
    # Nettoyage des NaN
    df_long = df_long.dropna(subset=['distance'])
    df_long['unique_id'] = df_long['site_name'] + "_" + df_long['transect_id']

    # 4. Interpolation et Lissage (indispensable pour les zones à cyclones)
    processed_list = []
    for tid in df_long['unique_id'].unique():
        subset = df_long[df_long['unique_id'] == tid].copy().sort_values('dates')
        if len(subset) > 10: # On ne garde que les transects avec assez de données
            subset['distance'] = subset['distance'].interpolate(method='linear')
            subset['distance'] = subset['distance'].rolling(window=3, center=True).mean()
            processed_list.append(subset.dropna())

    if not processed_list:
        print("Not enough data points after cleaning.")
        return
        
    df_final = pd.concat(processed_list)

    # 5. Normalisation avec le Scaler existant
    if os.path.exists(SCALER_FILE):
        with open(SCALER_FILE, 'rb') as f:
            scaler = pickle.load(f)
        df_final['distance_scaled'] = scaler.transform(df_final[['distance']])
    else:
        print("Warning: Scaler not found. Using raw distances.")

    # 6. Split Train/Test (80/20)
    # On split par transect pour s'assurer que l'IA teste sur des zones géographiques entières
    unique_ids = df_final['unique_id'].unique()
    train_ids, test_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
    
    df_test = df_final[df_final['unique_id'].isin(test_ids)]

    # Sauvegarde
    df_test.to_csv(OUTPUT_TEST_FILE, index=False)
    print(f"Success! Test set for {region_name} saved with {len(df_test)} points.")
    print(f"Number of test transects: {len(test_ids)}")

if __name__ == "__main__":
    prepare_test_region(TARGET_REGION)