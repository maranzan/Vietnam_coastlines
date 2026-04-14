import os
import sys
import ee
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

# --- CONFIGURATION ---
project_name = 'vietnam-coastal-erosion' 
ee.Initialize(project=project_name)

# Coordonnées Hội An
polygon = [[[108.36, 15.86], [108.40, 15.86], [108.40, 15.90], [108.36, 15.90], [108.36, 15.86]]]
polygon = SDS_tools.smallest_rectangle(polygon)

sitename = 'Hoi_An_IA'
filepath_data = os.path.join(os.getcwd(), 'data')

inputs = {
    'polygon': polygon,
    'dates': ['2020-01-01', '2025-01-01'],
    'sat_list': ['S2'], # Sentinel-2
    'sitename': sitename,
    'filepath': filepath_data,
}

metadata = SDS_download.retrieve_images(inputs)

settings = {
    'cloud_thresh': 0.1,
    'dist_clouds': 300,
    'output_epsg': 3857, # Projection standard
    'check_detection': False,
    'adjust_detection': False,
    'save_figure': True,
    'min_beach_area': 1000,
    'min_length_sl': 500,
    'cloud_mask_issue': False,
    'sand_color': 'default',
    'pan_off': False,
    's2cloudless_prob': 40, # key parameter for Sentinel-2 cloud masking (0-100, lower = more aggressive)
    'inputs': inputs,
}


output = SDS_shoreline.extract_shorelines(metadata, settings)

output = SDS_tools.remove_duplicates(output)
output = SDS_tools.remove_inaccurate_georef(output, 10)

print(f"Finished ! {len(output['shorelines'])} coastline exctracted.")