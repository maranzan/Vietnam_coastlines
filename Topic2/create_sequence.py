import pandas as pd
import numpy as np
import os
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
TEST_RATIO    = 0.15

print("Loading dataset...")
df = pd.read_csv('data/vietnam_master_multivariate.csv')

# 1. Reshape wide → long
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

# 2. Robust Outlier Removal (Ton code d'origine)
print("Removing outliers...")
def remove_outliers(group):
    group = group.copy()
    global_median = group['distance'].median()
    is_extreme    = (group['distance'] - global_median).abs() > 150
    group.loc[is_extreme, 'distance'] = np.nan
    
    rolling_median = group['distance'].rolling(window=5, center=True, min_periods=1).median()
    is_local       = (group['distance'] - rolling_median).abs() > 35
    group.loc[is_local, 'distance'] = np.nan
    
    group['distance'] = group['distance'].interpolate(method='linear').bfill().ffill()
    return group

df = df.sort_values(['unique_id', 'dates'])
df = df.groupby('unique_id', group_keys=False).apply(remove_outliers)
df = df.dropna(subset=feature_cols_base)

# ==========================================
# --- STEP 3: NEW GEOSPATIAL ENRICHMENT ---
# ==========================================
print("Injecting static geospatial features (Latitude, Longitude, Slope, Dist_River)...")

# Option A: Si tu as un CSV de géométrie, charge-lo ici :
# df_geo = pd.read_csv('data/transects_geometry.csv')
# df = pd.merge(df, df_geo, on='unique_id', how='left')

# Option B (Pour le test) : Simulation de la jointure si tu n'as pas encore le fichier
unique_transects = df['unique_id'].unique()
geo_mock = {
    'unique_id': unique_transects,
    'latitude': np.random.uniform(15.9, 16.5, len(unique_transects)),
    'longitude': np.random.uniform(108.2, 108.5, len(unique_transects)),
    'slope': np.random.uniform(0.01, 0.05, len(unique_transects)),
    'dist_river': np.random.uniform(100, 20000, len(unique_transects))
}
df_geo_mock = pd.DataFrame(geo_mock)
df = pd.merge(df, df_geo_mock, on='unique_id', how='left')

# Simulation de l'ajout du NDVI temporel (si pas déjà dans ton master CSV)
if 'NDVI' not in df.columns:
    df['NDVI'] = np.random.uniform(0.1, 0.6, len(df))

# ==========================================
# --- STEP 4: CALCULATING DERIVED METRICS ---
# ==========================================
print("Calculating shoreline changes and retreat rates...")

# A. Extraction du mois (Feature cyclique pour plus tard ou brute pour la classification)
df['month'] = df['dates'].dt.month

# B. Beach Width : C'est ta variable 'distance' actuelle (largeur de la plage visible)
df['beach_width'] = df['distance']

# C. Shoreline Change (Dshoreline)
df['shoreline_change'] = df.groupby('unique_id')['beach_width'].diff()

# D. Delta de temps en jours pour standardiser le Retreat Rate
df['time_delta_days'] = df.groupby('unique_id')['dates'].diff().dt.total_seconds() / (24 * 3600)

# E. Retreat Rate (m/day)
df['retreat_rate'] = df['shoreline_change'] / df['time_delta_days']

# Remplissage des NaN dus au calcul de différence (.diff())
df['shoreline_change'] = df['shoreline_change'].fillna(0)
df['retreat_rate'] = df['retreat_rate'].fillna(0)

# F. Hs_mean_7d : Calcul des vagues moyennes sur 7 jours
# Comme tes données ne sont pas forcément quotidiennes (dépend des passages satellites), 
# on fait une moyenne glissante temporelle basée sur les dates réelles.
df = df.sort_values(['unique_id', 'dates'])
df['Hs_mean_7d'] = df.groupby('unique_id')['wave_height'].transform(
    lambda x: x.rolling('7D', on=df.loc[x.index, 'dates'], min_periods=1).mean()
)

# ==========================================
# --- STEP 5: CREATING THE LABELS (CLASSES) ---
# ==========================================
print("Labeling erosion classes...")

def get_erosion_class(rate):
    # Seuils personnalisables (en mètres par jour de recul)
    if rate < -0.15:     # Plus de 15 cm de recul par jour = Forte érosion
        return 'high_erosion'
    elif rate < -0.02:   # Entre 2 et 15 cm de recul = Érosion modérée
        return 'moderate_erosion'
    elif rate <= 0.05:   # Variations minimes = Stable
        return 'stable'
    else:                # Avancée de la plage vers la mer = Accrétion
        return 'accretion'

df['erosion_class'] = df['retreat_rate'].apply(get_erosion_class)

# Encodage en label numérique pour l'IA
le = LabelEncoder()
df['label'] = le.fit_transform(df['erosion_class'])

# Save the encoder to decode predictions later
with open('data/label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

# ==========================================
# --- STEP 6: EXPORTING TABULAR DATA ---
# ==========================================
# Avant de faire du Seq2Seq, on sauvegarde ce magnifique dataset tabulaire.
# Il est parfait pour des modèles comme XGBoost, Random Forest ou LightGBM.

output_features = [
    'unique_id', 'latitude', 'longitude', 'month', 
    'shoreline_change', 'beach_width', 'slope', 
    'Hs_mean_7d', 'wave_period', 'tide_level', 'wind_speed', 
    'NDVI', 'dist_river', 'erosion_class', 'label'
]

spatial_df = df[output_features]
spatial_df.to_csv('data/vietnam_spatial_erosion_dataset.csv', index=False)
print(f"Tabular spatial dataset saved! Shape: {spatial_df.shape}")
print(spatial_df['erosion_class'].value_counts())
print("=" * 60)