import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- CONFIGURATION ---
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH    = 'model/erosion_seq2seq_v1.pth'
TEST_DATA     = 'data/dataset_ready_for_ia.csv'
SCALER_PATH   = 'data/scaler_seq2seq.pkl'
SEQ_LENGTH    = 10
PREDICT_STEPS = 36

# --- ARCHITECTURE ---
class Encoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)
    def forward(self, weather_future, h, c):
        out, _ = self.lstm(weather_future, (h, c))
        return self.fc(out)

class ErosionSeq2Seq(nn.Module):
    def __init__(self, enc_input=4, dec_input=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.encoder = Encoder(enc_input, hidden_size, num_layers)
        self.decoder = Decoder(dec_input, hidden_size, num_layers)
    def forward(self, x_enc, x_dec):
        h, c = self.encoder(x_enc)
        return self.decoder(x_dec, h, c)

# --- CHARGEMENT ---
for path in [SCALER_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        print(f"Error: {path} not found.")
        exit()

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

model = ErosionSeq2Seq().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- DONNÉES ---
df_test       = pd.read_csv(TEST_DATA, parse_dates=['dates'])
all_transects = df_test['unique_id'].unique()
target_id     = all_transects[0]
subset        = df_test[df_test['unique_id'] == target_id].sort_values('dates').reset_index(drop=True)

START_DATE = '2019-06-15'
start_idx  = (subset['dates'] - pd.to_datetime(START_DATE)).abs().idxmin()

if start_idx + SEQ_LENGTH + PREDICT_STEPS > len(subset):
    print(f"Error: not enough data after {START_DATE}.")
    exit()

feature_cols = ['distance', 'wave_height', 'wind_speed', 'wave_period']
all_features = subset[feature_cols].values

enc_block    = all_features[start_idx : start_idx + SEQ_LENGTH]
future_block = all_features[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]

dates_history = subset['dates'].iloc[start_idx : start_idx + SEQ_LENGTH]
dates_future  = subset['dates'].iloc[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]

print(f"History ends on : {dates_history.iloc[-1].strftime('%Y-%m-%d')}")
print(f"Forecast starts : {dates_future.iloc[0].strftime('%Y-%m-%d')}")

# --- HELPER ---
def inverse_distance(scaled_dist, weather_block):
    matrix        = np.zeros((len(scaled_dist), 4))
    matrix[:, 0]  = scaled_dist
    matrix[:, 1:] = weather_block
    return scaler.inverse_transform(matrix)[:, 0]

history_meters       = inverse_distance(enc_block[:, 0], enc_block[:, 1:])
real_observed_meters = inverse_distance(future_block[:, 0], future_block[:, 1:])
base_weather         = future_block[:, 1:].copy()  # (36, 3)

# --- PRÉDICTION : 36 mois en 1 seul forward pass ---
def predict_scenario(weather_forcing):
    # Construit le decoder input : dernière position connue + météo future
    last_pos = enc_block[-1, 0]  # dernière distance scalée connue
    dec_positions = np.concatenate([
        [[last_pos]],                          # premier pas : dernière position connue
        future_block[:-1, 0:1]                 # positions vraies décalées (pour scénario normal)
    ], axis=0)                                 # (36, 1)
    dec_input = np.concatenate([dec_positions, weather_forcing], axis=1)  # (36, 4)

    x_enc = torch.tensor(enc_block,  dtype=torch.float32).unsqueeze(0).to(device)
    x_dec = torch.tensor(dec_input,  dtype=torch.float32).unsqueeze(0).to(device)
    with torch.no_grad():
        preds = model(x_enc, x_dec).squeeze().cpu().numpy()
    return inverse_distance(preds, weather_forcing)

# --- SCÉNARIOS ---
weather_normal  = base_weather.copy()

weather_extreme = base_weather.copy()
weather_extreme[:, 0] = np.clip(weather_extreme[:, 0] * 1.50, 0, 1)
weather_extreme[:, 1] = np.clip(weather_extreme[:, 1] * 1.40, 0, 1)

weather_calm = base_weather.copy()
weather_calm[:, 0] = weather_calm[:, 0] * 0.50
weather_calm[:, 1] = weather_calm[:, 1] * 0.60

print("Running Seq2Seq predictions...")
meters_normal  = predict_scenario(weather_normal)
meters_extreme = predict_scenario(weather_extreme)
meters_calm    = predict_scenario(weather_calm)

# --- VISUALISATION ---
plt.figure(figsize=(14, 7))

plt.plot(dates_history, history_meters,
         label='Observed History (10 Input Pts)', color='black', marker='o', linewidth=2)
plt.plot(dates_future, real_observed_meters,
         label='True CoastSat Reality', color='purple', alpha=0.9, linewidth=2.5, marker='s')
plt.plot(dates_future, meters_normal,
         label='AI Scenario: Normal Weather', color='blue', linestyle='--', linewidth=2)
plt.plot(dates_future, meters_extreme,
         label='AI Scenario: Typhoon Stress Test', color='red', linestyle='--', linewidth=2)
plt.plot(dates_future, meters_calm,
         label='AI Scenario: Calm Sea Conditions', color='green', linestyle=':', linewidth=2)

plt.title(f'Seq2Seq 36-Month Forecast vs True CoastSat Reality — {target_id}')
plt.xlabel('Timeline (Dates)')
plt.ylabel('Shoreline Position (Meters)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- MÉTRIQUES ---
rmse_normal = np.sqrt(np.mean((meters_normal - real_observed_meters) ** 2))

print("\n" + "=" * 55)
print("SEQ2SEQ FORECAST RESULTS")
print("=" * 55)
print(f"Starting position       : {history_meters[-1]:.2f} m")
print(f"True CoastSat final     : {real_observed_meters[-1]:.2f} m")
print(f"Normal scenario final   : {meters_normal[-1]:.2f} m  (RMSE: {rmse_normal:.2f} m)")
print(f"Typhoon scenario final  : {meters_extreme[-1]:.2f} m")
print(f"Calm sea scenario final : {meters_calm[-1]:.2f} m")
print("=" * 55)