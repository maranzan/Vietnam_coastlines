import os
import geopandas as gpd
import pandas as pd

# --- 1. CONFIGURATION DES CHEMINS ---
# Mets ici le nom exact de ton fichier GeoJSON
GEOJSON_PATH = 'data/Da_Nang_My_Khe_transects.geojson' 
OUTPUT_CSV   = 'data/transects_geometry.csv'

# Note pour ton stage à l'HUMG : 
# Tu dois télécharger la couche des fleuves du Vietnam (format GeoJSON ou Shapefile)
# par exemple sur HydroSHEDS ou OpenStreetMap, et la placer ici :
RIVERS_GEOJSON = 'data/vietnam_rivers.geojson' 

print("Step 1: Loading CoastSat GeoJSON transects...")
# On charge ton fichier directement avec GeoPandas
gdf = gpd.read_file(GEOJSON_PATH)

# Au vu des coordonnées (ex: 12049973, 1813999), ton fichier est en EPSG:3857 (Web Mercator)
# On définit explicitement son système de coordonnées d'origine (CRS)
gdf.crs = "EPSG:3857"

# --- 2. CALCUL DU POINT CENTRAL ET CONVERSION LAT/LON ---
print("Step 2: Calculating real midpoint coordinates...")
# On extrait le centroïde (le milieu exact) de la ligne du transect
gdf['centroid_metric'] = gdf['geometry'].centroid

# Pour obtenir des Latitudes et Longitudes exploitables en degrés (WGS84 / EPSG:4326),
# on change le système de projection uniquement pour les points centraux
gdf_wgs84 = gdf.set_geometry('centroid_metric').to_crs(epsg=4326)

# Extraction propre des coordonnées réelles en degrés
gdf['latitude']  = gdf_wgs84['centroid_metric'].y
gdf['longitude'] = gdf_wgs84['centroid_metric'].x

# --- 3. CALCUL DE LA VRAIE DISTANCE AU FLEUVE ---
# Pour mesurer des distances réelles précises au Vietnam, on projette tout 
# dans le système local UTM zone 48N (EPSG:32648) qui utilise le mètre comme unité.
print("Step 3: Calculating distance to nearest river network...")
if os.path.exists(RIVERS_GEOJSON):
    gdf_rivers = gpd.read_file(RIVERS_GEOJSON)
    
    # Projection des transects et des rivières dans le même système métrique local
    gdf_metric = gdf.to_crs(epsg=32648)
    gdf_rivers_metric = gdf_rivers.to_crs(epsg=32648)
    
    # Pour chaque transect, on calcule sa distance minimale exacte avec la base de données des fleuves
    distances = []
    for geom in gdf_metric['geometry']:
        # .distance() renvoie la distance en mètres grâce à la projection 32648
        min_dist = gdf_rivers_metric.distance(geom).min()
        distances.append(round(min_dist, 2))
    gdf['dist_river'] = distances
else:
    print("⚠️ Warning: 'vietnam_rivers.geojson' not found. 'dist_river' set to 0.0.")
    gdf['dist_river'] = 0.0

# --- 4. ASSIGNATION DES PENTES RÉELLES (SLOPE) ---
# Idéalement, la pente provient du Modèle Numérique de Terrain (DEM) fourni par tes professeurs.
# En attendant ta matrice finale, on applique la valeur par défaut standard (0.02)
# mais issue d'une vraie colonne si elle existe dans ton dictionnaire.
gdf['slope'] = gdf['slope_measured'] if 'slope_measured' in gdf.columns else 0.02

# --- 5. RE-FORMATAGE ET EXPORTATION FINALE ---
print("Step 4: Mapping IDs and exporting...")
# Ton fichier contient une propriété 'id' (TS_001, TS_002...)
# On crée le 'unique_id' au format : Da_Nang_My_Khe_TS_001
site_prefix = "Da_Nang_My_Khe"
gdf['unique_id'] = site_prefix + "_" + gdf['id']

# Sélection des colonnes requises, sans aucune valeur simulée au hasard
final_df = gdf[['unique_id', 'latitude', 'longitude', 'slope', 'dist_river']]

# Sauvegarde finale sous format CSV
os.makedirs('data', exist_ok=True)
final_df.to_csv(OUTPUT_CSV, index=False)

print("=" * 60)
print(f"Geometry File Successfully Created: {OUTPUT_CSV}")
print(f"Processed {len(final_df)} real coastal transects.")
print("=" * 60)
print(final_df.head())