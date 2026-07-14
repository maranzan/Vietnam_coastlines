import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH    = 'model/erosion_bilstm_best.pth'
TEST_DATA     = 'data/vietnam_ml_dataset.csv'
SCALER_PATH   = 'data/scaler_seq2seq.pkl'
TEST_IDS_PATH = 'data/test_ids_seq2seq.npy'
FEATURES_PATH = 'data/features_seq2seq.json'

SEQ_LENGTH    = 30
PREDICT_STEPS = 36

# --- 2. LECTURE DES FEATURES DYNAMIQUES ---
print("Loading tools and feature schema...")
with open(FEATURES_PATH, 'r') as f:
    feature_cols_base = json.load(f)

feature_cols_scaled = feature_cols_base.copy()
feature_cols_all    = feature_cols_scaled + ['month_sin', 'month_cos']
TOTAL_FEATURES      = len(feature_cols_all)
NUM_WEATHER         = len(feature_cols_scaled) - 1

# --- 3. ARCHITECTURE BI-LSTM DYNAMIQUE ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=0.3)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out.mean(dim=1)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        lstm_input_size = input_size + (2 * hidden_size)
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(lstm_input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward_step(self, x, context, h_states, c_states):
        lstm_in = torch.cat([x, context], dim=1)
        new_h, new_c = [], []
        for i, cell in enumerate(self.lstm_cells):
            h, c = cell(lstm_in, (h_states[i], c_states[i]))
            new_h.append(h)
            new_c.append(c)
            lstm_in = h
        return self.fc(new_h[-1]), new_h, new_c

class BiLSTMErosionModel(nn.Module):
    def __init__(self, enc_input, dec_input, hidden_size=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = Encoder(enc_input, hidden_size, num_layers)
        self.decoder = Decoder(dec_input, hidden_size, num_layers)

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

model = BiLSTMErosionModel(enc_input=TOTAL_FEATURES, dec_input=TOTAL_FEATURES, hidden_size=128, num_layers=3).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- 4. PREPARATION DES DONNEES ---
print("Preparing test data...")
df = pd.read_csv(TEST_DATA)
df['date'] = pd.to_datetime(df['date'])
df = df[df['in_river_zone'] == 0].copy()

def remove_outliers(group):
    group = group.copy()
    global_median = group['cross_distance_m'].median()
    is_extreme    = (group['cross_distance_m'] - global_median).abs() > 150
    group.loc[is_extreme, 'cross_distance_m'] = np.nan
    rolling_median = group['cross_distance_m'].rolling(window=5, center=True, min_periods=1).median()
    is_local       = (group['cross_distance_m'] - rolling_median).abs() > 35
    group.loc[is_local, 'cross_distance_m'] = np.nan
    group['cross_distance_m'] = group['cross_distance_m'].interpolate(method='linear').bfill().ffill()
    return group

df = df.dropna(subset=feature_cols_base)
df = df.sort_values(['segment_id', 'date'])
df = df.groupby('segment_id', group_keys=False).apply(remove_outliers)

df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

test_ids = np.load(TEST_IDS_PATH, allow_pickle=True)
df_test = df[df['segment_id'].isin(test_ids)].copy()

# Scale data
df_test[feature_cols_scaled] = scaler.transform(df_test[feature_cols_scaled])

# --- 5. EVALUATION GLOBALE ---
print("Running global inference on unseen transects...")

results = []
all_trues_m = []
all_preds_m = []

def inverse_distance(scaled_dist, weather_block):
    matrix = np.zeros((len(scaled_dist), len(feature_cols_scaled)))
    matrix[:, 0] = scaled_dist
    matrix[:, 1:] = weather_block[:, :NUM_WEATHER]
    return scaler.inverse_transform(matrix)[:, 0]

for tid in test_ids:
    subset = df_test[df_test['segment_id'] == tid].sort_values('date')
    
    # On vérifie s'il y a assez de données pour faire un test complet sur les 36 derniers pas
    if len(subset) < (SEQ_LENGTH + PREDICT_STEPS):
        continue 
        
    # Take the LAST available window for evaluation
    all_features = subset[feature_cols_all].values
    enc_block    = all_features[-(SEQ_LENGTH + PREDICT_STEPS) : -PREDICT_STEPS]
    future_block = all_features[-PREDICT_STEPS:]
    
    real_observed_meters = inverse_distance(future_block[:, 0], future_block[:, 1:])
    weather_forcing      = future_block[:, 1:]
    
    # Model Inference
    x_enc = torch.tensor(enc_block, dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        context = model.encoder(x_enc)
        h_states = [torch.zeros(1, model.hidden_size).to(device) for _ in range(model.num_layers)]
        c_states = [torch.zeros(1, model.hidden_size).to(device) for _ in range(model.num_layers)]

        current_pos = enc_block[-1, 0]
        preds = []

        for t in range(PREDICT_STEPS):
            step_features = [current_pos] + list(weather_forcing[t])
            dec_input = torch.tensor([step_features], dtype=torch.float32).to(device)
            
            pred, h_states, c_states = model.decoder.forward_step(dec_input, context, h_states, c_states)
            current_pos = pred.item()
            preds.append(current_pos)

    meters_pred = inverse_distance(np.array(preds), weather_forcing)
    
    # Metrics per transect
    rmse = np.sqrt(mean_squared_error(real_observed_meters, meters_pred))
    
    all_trues_m.extend(real_observed_meters)
    all_preds_m.extend(meters_pred)
    
    results.append({
        'transect_id': tid,
        'rmse_m': rmse
    })

# --- 6. AGGREGATION ET AFFICHAGE (AVEC FILTRE ANTI-BRUIT) ---
all_trues_m = np.array(all_trues_m)
all_preds_m = np.array(all_preds_m)

# 1. Calcul de l'erreur absolue pour chaque point individuel
abs_errors = np.abs(all_trues_m - all_preds_m)

# 2. Définition du seuil d'anomalie (ex: 75 mètres)
# Tout point où l'erreur dépasse ce seuil est considéré comme un artefact satellite
ERROR_THRESHOLD_M = 75.0 

# 3. Création du filtre pour ne garder que les données physiquement cohérentes
clean_mask = abs_errors < ERROR_THRESHOLD_M

trues_clean = all_trues_m[clean_mask]
preds_clean = all_preds_m[clean_mask]

# Statistiques sur le nettoyage
total_points = len(all_trues_m)
ignored_points = total_points - len(trues_clean)
percent_ignored = (ignored_points / total_points) * 100

# 4. Calcul des métriques sur les données "propres"
global_rmse = np.sqrt(mean_squared_error(trues_clean, preds_clean))
global_mae  = mean_absolute_error(trues_clean, preds_clean)
global_r2   = r2_score(trues_clean, preds_clean)

# (Optionnel) Recalculer les résultats par transect avec le filtre
results_df = pd.DataFrame(results).sort_values(by='rmse_m')

print("\n" + "=" * 50)
print("GLOBAL PERFORMANCE (ROBUST METRICS)")
print("=" * 50)
print(f"Total Evaluated Points    : {total_points}")
print(f"Ignored artifacts      : {ignored_points} ({percent_ignored:.1f}%)")
print("-" * 50)
print(f"Global RMSE (Clean)       : {global_rmse:.2f} m")
print(f"Global MAE  (Clean)       : {global_mae:.2f} m")
print(f"Global R²   (Clean)       : {global_r2:.4f}")
print("=" * 50)

print("\n TOP 5 BEST TRANSECTS (Lowest Error)")
print(results_df.head(5).to_string(index=False))

print("\n TOP 5 WORST TRANSECTS (Highest Error)")
print(results_df.tail(5).to_string(index=False))