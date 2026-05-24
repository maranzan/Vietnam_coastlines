import os
import pickle
import pandas as pd
import numpy as np
import warnings
import sys
from coastsat import SDS_transects
from shapely.geometry import LineString
from concurrent.futures import ProcessPoolExecutor

# --- WARNING MANAGEMENT ---
# Ignore fragmentation and runtime warnings
warnings.filterwarnings('ignore')
np.seterr(all='ignore')

DATA_ROOT = os.path.join(os.getcwd(), 'data')
OUTPUT_MASTER_FILE = 'data/vietnam_master_dataset.csv'
SPACING = 100 
TRANSECT_LENGTH = 200

def process_one_site(site_name):
    # Redirect stderr to null to silence CoastSat internal warnings
    original_stderr = sys.stderr
    sys.stderr = open(os.devnull, 'w')
    
    try:
        site_path = os.path.join(DATA_ROOT, site_name)
        checkpoint_file = os.path.join(site_path, f'{site_name}_processed_distances.csv')
        pkl_file = os.path.join(site_path, f'{site_name}_output.pkl')
        
        # 1. Check for existing cache
        if os.path.exists(checkpoint_file):
            print(f"SITE: {site_name} | Status: Already processed (loading cache)")
            return pd.read_csv(checkpoint_file)

        if not os.path.exists(pkl_file):
            return None

        with open(pkl_file, 'rb') as f:
            output = pickle.load(f)

        # 2. Transect generation
        reference_sl = output['shorelines'][0]
        line = LineString(reference_sl)
        distances = np.arange(0, line.length, SPACING)
        transects = {}
        
        for i, d in enumerate(distances):
            p1 = line.interpolate(d)
            p2 = line.interpolate(d + 0.1)
            dx, dy = p2.x - p1.x, p2.y - p1.y
            mag = np.sqrt(dx**2 + dy**2)
            ux, uy = -dy/mag, dx/mag
            start = [p1.x - ux * (TRANSECT_LENGTH/2), p1.y - uy * (TRANSECT_LENGTH/2)]
            end = [p1.x + ux * (TRANSECT_LENGTH/2), p1.y + uy * (TRANSECT_LENGTH/2)]
            transects[f'TS_{i+1:03d}'] = np.array([start, end])
        
        # 3. Progress report
        n_dates = len(output['dates'])
        n_ts = len(transects)
        print(f"RUNNING: {site_name} | Complexity: {n_ts} transects x {n_dates} dates")

        settings_transects = {
            'along_dist': 25, 'min_points': 3, 'max_std': 15,
            'max_range': 30, 'min_chainage': -100, 'multiple_inter': 'auto',
            'auto_prc': 0.1
        }
        
        # Heavy geometrical computation
        cross_distance = SDS_transects.compute_intersection_QC(output, transects, settings_transects)

        # 4. Data structuring (Dictionary approach to avoid fragmentation)
        data_dict = {'dates': output['dates'], 'site_name': site_name}
        for ts_name, dists in cross_distance.items():
            data_dict[ts_name] = dists
        
        df_site = pd.DataFrame(data_dict)
        
        # Save individual site cache
        df_site.to_csv(checkpoint_file, index=False)
        print(f"DONE: {site_name} | Status: Saved to CSV")
        return df_site

    except Exception as e:
        print(f"ERROR: {site_name} | Message: {str(e)}")
        return None
    finally:
        # Restore stderr
        sys.stderr.close()
        sys.stderr = original_stderr

if __name__ == '__main__':
    all_sites = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    print("-" * 50)
    print(f"SYSTEM: {os.cpu_count()} processors detected")
    print(f"INPUT: {len(all_sites)} sites found in data folder")
    print("CONFIG: Silent mode enabled (NaN warnings hidden)")
    print("-" * 50)

    # Launch parallel processing using all but one core
    with ProcessPoolExecutor(max_workers=os.cpu_count() - 1) as executor:
        results = list(executor.map(process_one_site, all_sites))

    # Filter out None results and merge
    valid_results = [res for res in results if res is not None]
    
    if valid_results:
        print("\nMERGING: Compiling final master dataset...")
        master_df = pd.concat(valid_results, axis=0, ignore_index=True)
        
        # Reorder columns to have dates and site_name first
        cols = ['dates', 'site_name'] + [c for c in master_df.columns if c not in ['dates', 'site_name']]
        master_df = master_df[cols]
        
        master_df.to_csv(OUTPUT_MASTER_FILE, index=False)
        print(f"SUCCESS: {OUTPUT_MASTER_FILE} created with {len(master_df)} total rows")
    else:
        print("WARNING: No data was processed successfully")