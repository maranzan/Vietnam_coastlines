import os
import pickle
import pandas as pd
import numpy as np
from coastsat import SDS_transects
from shapely.geometry import LineString

data_root = os.path.join(os.getcwd(), 'data')
output_master_file = 'vietnam_master_dataset.csv'
spacing = 50 
transect_length = 200

all_site_data = []

sites = [d for d in os.listdir(data_root) if os.path.isdir(os.path.join(data_root, d))]

for site in sites:
    print(f"\nTreating site : {site}")
    site_path = os.path.join(data_root, site)
    pkl_file = os.path.join(site_path, f'{site}_output.pkl')
    
    if not os.path.exists(pkl_file):
        print(f"File {site}_output.pkl not_found for {site}. Skipping...")
        continue

    with open(pkl_file, 'rb') as f:
        output = pickle.load(f)

    print(f"Generating transects for {site}...")
    reference_sl = output['shorelines'][0]
    line = LineString(reference_sl)
    distances = np.arange(0, line.length, spacing)
    transects = {}
    
    for i, d in enumerate(distances):
        p1 = line.interpolate(d)
        p2 = line.interpolate(d + 0.1)
        dx, dy = p2.x - p1.x, p2.y - p1.y
        mag = np.sqrt(dx**2 + dy**2)
        ux, uy = -dy/mag, dx/mag
        start = [p1.x - ux * (transect_length/2), p1.y - uy * (transect_length/2)]
        end = [p1.x + ux * (transect_length/2), p1.y + uy * (transect_length/2)]
        transects[f'TS_{i+1:03d}'] = np.array([start, end])

    settings_transects = {
        'along_dist': 25, 'min_points': 3, 'max_std': 15,
        'max_range': 30, 'min_chainage': -100, 'multiple_inter': 'auto'
    }
    cross_distance = SDS_transects.compute_intersection_QC(output, transects, settings_transects)

    df_site = pd.DataFrame({'dates': output['dates']})
    df_site['site_name'] = site
    
    for ts_name, distances in cross_distance.items():
        df_site[ts_name] = distances
    
    all_site_data.append(df_site)

if all_site_data:
    master_df = pd.concat(all_site_data, axis=0, ignore_index=True)
    cols = ['dates', 'site_name'] + [c for c in master_df.columns if c not in ['dates', 'site_name']]
    master_df = master_df[cols]
    
    master_df.to_csv(output_master_file, index=False)
    print(f"\nMASTER CSV CREATED : {output_master_file}")
