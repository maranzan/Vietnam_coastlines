import os
import ee
from coastsat import SDS_download, SDS_preprocess, SDS_shoreline, SDS_tools, SDS_transects

project_name = 'vietnam-coastal-erosion' 
ee.Initialize(project=project_name)

SITES = {
    'Hoi_An_IA': [[[108.36, 15.86], [108.40, 15.86], [108.40, 15.90], [108.36, 15.90], [108.36, 15.86]]],
    'Da_Nang_My_Khe': [[[108.24, 16.05], [108.27, 16.05], [108.27, 16.08], [108.24, 16.08], [108.24, 16.05]]],
    'Mui_Ne': [[[108.18, 10.92], [108.22, 10.92], [108.22, 10.95], [108.18, 10.95], [108.18, 10.92]]],
    'QN_Dien_Ban': [[[108.30, 15.88], [108.36, 15.88], [108.36, 15.95], [108.30, 15.95], [108.30, 15.88]]],
    'QN_Thang_Binh': [[[108.40, 15.65], [108.52, 15.65], [108.52, 15.85], [108.40, 15.85], [108.40, 15.65]]],
    'QN_Tam_Thanh': [[[108.50, 15.50], [108.62, 15.50], [108.62, 15.68], [108.50, 15.68], [108.50, 15.50]]],
    'QN_Nui_Thanh': [[[108.60, 15.40], [108.75, 15.40], [108.75, 15.55], [108.60, 15.55], [108.60, 15.40]]]}

filepath_data = os.path.join(os.getcwd(), 'data')

for sitename, poly_coords in SITES.items():
    print(sitename)
    
    polygon = SDS_tools.smallest_rectangle(poly_coords)

    inputs = {
        'polygon': polygon,
        'dates': ['2018-01-01', '2025-01-01'],
        'sat_list': ['S2'], 
        'sitename': sitename,
        'filepath': filepath_data,
    }

    metadata = SDS_download.retrieve_images(inputs)

    settings = {
        'cloud_thresh': 0.1,
        'dist_clouds': 300,
        'output_epsg': 3857, 
        'check_detection': False,
        'adjust_detection': False,
        'save_figure': True,
        'min_beach_area': 1000,
        'min_length_sl': 500,
        'cloud_mask_issue': False,
        'sand_color': 'default',
        'pan_off': False,
        's2cloudless_prob': 40,
        'inputs': inputs,
    }

    output = SDS_shoreline.extract_shorelines(metadata, settings)
    
    if len(output['shorelines']) > 0:
        output = SDS_tools.remove_duplicates(output)
        output = SDS_tools.remove_inaccurate_georef(output, 10)
        print(f"{sitename} done : {len(output['shorelines'])} lignes extracted.")
    else:
        print(f"no line detected for {sitename}.")