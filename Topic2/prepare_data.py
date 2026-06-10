import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore')

print("1. Chargement des données brutes...")
df = pd.read_csv('data/vietnam_ml_dataset.csv')

# Exclure les zones d'embouchure (trop chaotiques)
df = df[df['in_river_zone'] == 0].copy()
df['date'] = pd.to_datetime(df['date'])
df['YearMonth'] = df['date'].dt.to_period('M')

print("2. Agrégation mensuelle des données...")
aggregation_rules = {
    'site_name': 'first',
    'latitude': 'first',
    'longitude': 'first',
    'orientation_deg': 'first',
    'slope_deg': 'first',
    'elevation_m': 'first',
    'dist_river_m': 'first',
    'cross_distance_m': 'mean', # Position moyenne sur le mois
    'Hs_mean_7d': 'mean',
    'Hs_max_7d': 'mean',
    'wave_period_s': 'mean', 
    'wave_dir_deg': 'mean',
    'wind_mean_7d': 'mean',
    'wind_dir_deg': 'mean'
}

existing_cols = {k: v for k, v in aggregation_rules.items() if k in df.columns}
existing_cols['cross_distance_m'] = 'mean'

# Groupement par mois et par transect
df_monthly = df.groupby(['segment_id', 'YearMonth']).agg(existing_cols).reset_index()
df_monthly = df_monthly.sort_values(by=['segment_id', 'YearMonth'])

print("3. Calcul de l'évolution (Target) et filtrage physique...")
target = 'monthly_change_m'
df_monthly[target] = df_monthly.groupby('segment_id')['cross_distance_m'].diff()

df_monthly = df_monthly.dropna(subset=[target])

# --- LE FILTRE PHYSIQUE ---
# On supprime les variations mensuelles supérieures à 15 mètres 
# (ce sont quasi systématiquement des erreurs de détection du satellite)
max_monthly_change = 15.0 
df_monthly = df_monthly[
    (df_monthly[target] >= -max_monthly_change) & 
    (df_monthly[target] <= max_monthly_change)
]

output_file = 'data/vietnam_monthly_dataset.csv'
df_monthly.to_csv(output_file, index=False)
print(f"\n[SUCCÈS] Fichier mensuel généré : {output_file}")
print(f"Nombre d'observations conservées : {len(df_monthly)}")