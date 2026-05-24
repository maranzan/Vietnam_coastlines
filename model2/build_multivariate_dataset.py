import cdsapi
import os
import xarray as xr
import glob
import pandas as pd
import numpy as np

# --- CONFIGURATION ---
DATA_ROOT = 'data'
ENV_CSV = os.path.join(DATA_ROOT, 'era5_tide_vietnam.csv')
MASTER_OUTPUT = os.path.join(DATA_ROOT, 'vietnam_master_multivariate.csv')
# Quang Nam coordinates
AREA = [16.5, 108.0, 15.0, 109.5] # North, West, South, East

# ==========================================================
# STEP 1: DOWNLOAD ERA5 WEATHER DATA (YEAR BY YEAR)
# ==========================================================
def download_weather_data():
    c = cdsapi.Client()
    years = [str(y) for y in range(2018, 2025)]
    
    print("--- Phase 1: Downloading Copernicus ERA5 Data ---")
    for year in years:
        file_path = os.path.join(DATA_ROOT, f'era5_{year}.nc')
        if not os.path.exists(file_path):
            print(f"Requesting data for {year}...")
            try:
                c.retrieve('reanalysis-era5-single-levels', {
                    'product_type': 'reanalysis',
                    'format': 'netcdf',
                    'variable': [
                        'significant_height_of_combined_wind_waves_and_swell',
                        '10m_u_component_of_wind', 
                        '10m_v_component_of_wind',
                        'mean_wave_period'
                    ],
                    'year': year,
                    'month': [f"{m:02d}" for m in range(1, 13)],
                    'day': [f"{d:02d}" for d in range(1, 32)],
                    'time': [f"{h:02d}:00" for h in range(0, 24, 3)],
                    'area': AREA,
                }, file_path)
            except Exception as e:
                print(f"Failed to download {year}: {e}")
        else:
            print(f"Year {year} already exists in {DATA_ROOT}.")

# ==========================================================
# STEP 2: COMPILE NETCDF TO CLEAN CSV
# ==========================================================
import zipfile
import glob
import os
import xarray as xr
import pandas as pd
import numpy as np

def compile_environmental_csv():
    print("\n--- Phase 2: Compiling Weather Files ---")
    
    # 1. Extraction des ZIPs (Copernicus CDS-Beta envoie souvent des .nc qui sont en fait des .zip)
    zip_files = glob.glob(os.path.join(DATA_ROOT, 'era5_*.nc'))
    for f in zip_files:
        if zipfile.is_zipfile(f):
            year = f.split('_')[-1].replace('.nc', '')
            print(f"Extracting zip content for year {year}...")
            with zipfile.ZipFile(f, 'r') as zip_ref:
                for member in zip_ref.namelist():
                    filename = os.path.basename(member)
                    source = zip_ref.open(member)
                    # On renomme pour éviter que data.nc n'écrase les autres années
                    target_path = os.path.join(DATA_ROOT, f"streams/extracted_{year}_{filename}")
                    with open(target_path, "wb") as target:
                        target.write(source.read())
    
    # 2. Récupération des vrais fichiers NetCDF extraits
    valid_nc = glob.glob(os.path.join(DATA_ROOT, "streams/extracted_*.nc"))
    if not valid_nc:
        print("Error: No valid NetCDF files found in 'data/' folder.")
        return

    print(f"Merging {len(valid_nc)} NetCDF files...")
    # On force engine='netcdf4' pour la lecture des fichiers extraits
    ds = xr.open_mfdataset(valid_nc, combine='by_coords', engine='netcdf4')
    
    # 3. Sélection des variables d'intérêt AVANT conversion en DataFrame
    # swh: wave height, mwp: wave period, u10/v10: wind components
    vars_to_keep = [v for v in ['swh', 'mwp', 'u10', 'v10'] if v in ds.data_vars]
    
    # Détection de la colonne temps (peut être 'time' ou 'valid_time')
    time_col = 'time' if 'time' in ds.coords else 'valid_time'
    
    # Conversion propre en filtrant les métadonnées inutiles
    df = ds[vars_to_keep].to_dataframe().reset_index()
    
    # Harmonisation du nom de la colonne temps
    df = df.rename(columns={time_col: 'datetime'})
    
    # 4. Calcul de la moyenne spatiale (agrégation par date)
    # numeric_only=True empêche Pandas de concaténer des strings
    df = df.groupby('datetime').mean(numeric_only=True).reset_index()
    
    # 5. Renommage final et calcul du vent
    rename_dict = {
        'swh': 'wave_height',
        'mwp': 'wave_period'
    }
    df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
    
    if 'u10' in df.columns and 'v10' in df.columns:
        df['wind_speed'] = np.sqrt(df['u10']**2 + df['v10']**2)
    
    # Ajout de la colonne marée (placeholder)
    df['tide_level'] = 0.0 
    
    # Sélection des colonnes finales
    final_cols = ['datetime', 'wave_height', 'wave_period', 'wind_speed', 'tide_level']
    # On ne garde que les colonnes qui existent réellement
    df = df[[c for c in final_cols if c in df.columns]]
    
    # Sauvegarde
    df.to_csv(ENV_CSV, index=False)
    print(f"Success: Environmental data saved to {ENV_CSV}")
    print(f"Total timestamps processed: {len(df)}")

# N'oublie pas de définir DATA_ROOT et ENV_CSV en haut de ton script si ce n'est pas fait

# ==========================================================
# STEP 3: MERGE WITH EXISTING TRANSECT DISTANCES
# ==========================================================
def merge_with_transects():
    print("\n--- Phase 3: Merging with Site Distances ---")
    if not os.path.exists(ENV_CSV):
        print("Error: Environmental CSV missing.")
        return

    # Charger les données météo
    env_df = pd.read_csv(ENV_CSV, parse_dates=['datetime']).sort_values('datetime')
    
    # --- LA CORRECTION ICI ---
    # On enlève le fuseau horaire (UTC) pour qu'il soit compatible avec CoastSat
    if env_df['datetime'].dt.tz is not None:
        env_df['datetime'] = env_df['datetime'].dt.tz_localize(None)
    # --------------------------

    all_site_data = []
    
    sites = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    for site in sites:
        # Chemin vers ton fichier de distances généré par CoastSat
        csv_path = os.path.join(DATA_ROOT, site, f'{site}_processed_distances.csv')
        
        if os.path.exists(csv_path):
            print(f"Merging site: {site}")
            # Chargement des distances du site
            df_site = pd.read_csv(csv_path, parse_dates=['dates']).sort_values('dates')
            
            # Nettoyage des dates CoastSat (au cas où elles auraient aussi un fuseau horaire)
            if df_site['dates'].dt.tz is not None:
                df_site['dates'] = df_site['dates'].dt.tz_localize(None)
            
            # La jointure magique : merge_asof
            # Il cherche la météo la plus proche AVANT (backward) la date de l'image
            merged = pd.merge_asof(
                df_site, 
                env_df, 
                left_on='dates', 
                right_on='datetime', 
                direction='backward'
            )
            
            # On ajoute le nom du site pour pouvoir différencier les transects plus tard
            merged['site_name'] = site
            
            # Optionnel : supprimer la colonne datetime redondante (on garde 'dates')
            if 'datetime' in merged.columns:
                merged = merged.drop(columns=['datetime'])
                
            all_site_data.append(merged)
        else:
            print(f"Skipping {site}: no processed_distances.csv found.")

    if all_site_data:
        # Fusion de tous les sites en un seul DataFrame
        final_df = pd.concat(all_site_data, axis=0, ignore_index=True)
        
        # Création d'un ID unique par transect (site + nom du transect)
        # Indispensable pour ton script de création de séquences
        # (On suppose que tes colonnes de transects commencent par 'TS')
        
        # Sauvegarde du dataset final prêt pour l'IA
        final_df.to_csv(MASTER_OUTPUT, index=False)
        print(f"\nDONE! Dataset multivarié créé : {MASTER_OUTPUT}")
        print(f"Dimensions finales : {final_df.shape}")
    else:
        print("Error: No site data was merged.")

# --- EXECUTION ---
if __name__ == "__main__":
    os.makedirs(DATA_ROOT, exist_ok=True)
    download_weather_data()
    compile_environmental_csv()
    merge_with_transects()