import os
import sys
import ee
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

# --- CONFIGURATION ---
project_name = 'vietnam-coastal-erosion' # Ton projet Google Cloud
ee.Initialize(project=project_name)

# Coordonnées Hội An
polygon = [[[108.36, 15.86], [108.40, 15.86], [108.40, 15.90], [108.36, 15.90], [108.36, 15.86]]]
polygon = SDS_tools.smallest_rectangle(polygon)

sitename = 'Hoi_An_IA'
filepath_data = os.path.join(os.getcwd(), 'data')

inputs = {
    'polygon': polygon,
    'dates': ['2020-01-01', '2025-01-01'],
    'sat_list': ['S2'], # On reste sur Sentinel-2 pour la précision (10m)
    'sitename': sitename,
    'filepath': filepath_data,
}

# --- ÉTAPE 1 : RÉCUPÉRATION ---
metadata = SDS_download.retrieve_images(inputs)

# --- ÉTAPE 2 : PARAMÈTRES IA & NETTOYAGE ---
settings = {
    'cloud_thresh': 0.1,
    'dist_clouds': 300,
    'output_epsg': 3857, # Projection standard (Web Mercator)
    'check_detection': False,
    'adjust_detection': False,
    'save_figure': True,
    'min_beach_area': 1000,
    'min_length_sl': 500,
    'cloud_mask_issue': False,
    'sand_color': 'default',
    'pan_off': False,
    's2cloudless_prob': 40, # La clé qui posait problème (on la définit ici)
    'inputs': inputs,
}

# --- ÉTAPE 3 : EXTRACTION (Le coeur du projet) ---
# Cette étape crée les données d'entraînement pour ton IA
output = SDS_shoreline.extract_shorelines(metadata, settings)

# Nettoyage des résultats
output = SDS_tools.remove_duplicates(output)
output = SDS_tools.remove_inaccurate_georef(output, 10)

print(f"✅ Terminé ! {len(output['shorelines'])} lignes de côte extraites.")