import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import geopandas as gpd
import folium
import pickle
import os
from shapely.geometry import LineString

class ErosionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(ErosionLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

device = torch.device('cpu')
model = ErosionLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
model.load_state_dict(torch.load('model/erosion_lstm_v1.pth', map_location=device))
model.eval()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

site = 'Mui_Ne'
path_geojson = f'data/{site}/{site}_transects.geojson'

gdf = gpd.read_file(path_geojson)

gdf.crs = "EPSG:3857" 

gdf = gdf.to_crs(epsg=4326)

print(f"Coordonnées corrigées : {gdf.geometry.iloc[0].coords[0]}")

df_master = pd.read_csv('vietnam_master_dataset.csv')
df_site = df_master[df_master['site_name'] == site].sort_values('dates')

predictions = []
transect_cols = [c for c in df_site.columns if c.startswith('TS_')]

for col in transect_cols:
    series = df_site[col].dropna().values
    if len(series) >= 10:
        scaled = scaler.transform(series[-10:].reshape(-1, 1))
        inp = torch.from_numpy(scaled).float().view(1, 10, 1)
        with torch.no_grad():
            pred = scaler.inverse_transform(model(inp).numpy())[0][0]
        predictions.append({'id': col, 'diff': pred - series[-1], 'current': series[-1], 'pred': pred})

df_res = pd.DataFrame(predictions)
gdf = gdf.merge(df_res, on='id')

m = folium.Map(location=[gdf.geometry.centroid.y.mean(), gdf.geometry.centroid.x.mean()], zoom_start=15)
folium.TileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', 
                 attr='Esri', name='Satellite').add_to(m)

for _, row in gdf.iterrows():
    color = 'red' if row['diff'] < -0.5 else 'green' if row['diff'] > 0.5 else 'orange'
    
    points = [(p[1], p[0]) for p in row.geometry.coords]
    folium.PolyLine(points, color=color, weight=5, opacity=0.8,
                    popup=f"Transect: {row['id']}<br>Evolution: {row['diff']:.2f}m").add_to(m)

m.save(f'data/{site}_erosion_map.html')
print(f"Finished. Open 'data/{site}_erosion_map.html'")