import os
import pickle
import numpy as np
from shapely.geometry import LineString, Point


sitename = 'Hoi_An_IA'
filepath = os.path.join(os.getcwd(), 'data', sitename, f'{sitename}_output.pkl')
with open(filepath, 'rb') as f:
    output = pickle.load(f)


reference_sl = output['shorelines'][0]
line = LineString(reference_sl)

spacing = 100
transect_length = 200
transects = {}

distances = np.arange(0, line.length, spacing)
for i, d in enumerate(distances):

    p1 = line.interpolate(d)
    p2 = line.interpolate(d + 0.1)

    dx = p2.x - p1.x
    dy = p2.y - p1.y
    mag = np.sqrt(dx**2 + dy**2)
    ux, uy = -dy/mag, dx/mag
    
    start_point = [p1.x - ux * (transect_length/2), p1.y - uy * (transect_length/2)]
    end_point = [p1.x + ux * (transect_length/2), p1.y + uy * (transect_length/2)]
    
    transects[f'TS_{i+1:03d}'] = np.array([start_point, end_point])

with open(os.path.join(os.getcwd(), 'data', sitename, 'transects_auto.pkl'), 'wb') as f:
    pickle.dump(transects, f)

print(f"{len(transects)} transects generated and saved to transects_auto.pkl")