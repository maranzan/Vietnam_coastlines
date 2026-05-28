import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- CONFIGURATION ---
device        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH    = 'model/erosion_seq2seq_v3.pth'
TEST_DATA     = 'data/vietnam_master_multivariate.csv'
SCALER_PATH   = 'data/scaler_seq2seq.pkl'

SEQ_LENGTH    = 10
PREDICT_STEPS = 36
TRANSECT_IDX  = 1200      # Change this index to test other unseen beaches
START_DATE    = '2020-06-15'

# --- ARCHITECTURE V3 (7 FEATURES) ---
class Encoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
    def forward(self, x):
        _, (h, c) = self.lstm(x)
        return h, c

class Decoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm        = nn.LSTMCell(input_size, hidden_size)
        self.lstm2       = nn.LSTMCell(hidden_size, hidden_size) if num_layers > 1 else None
        self.fc          = nn.Linear(hidden_size, output_size)

    def forward_step(self, x, h1, c1, h2, c2):
        h1, c1 = self.lstm(x, (h1, c1))
        if self.lstm2 is not None:
            h2, c2 = self.lstm2(h1, (h2, c2))
            out = self.fc(h2)
        else:
            out = self.fc(h1)
        return out, h1, c1, h2, c2

class ErosionSeq2Seq(nn.Module):
    def __init__(self, enc_input=7, dec_input=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.encoder = Encoder(enc_input, hidden_size, num_layers)
        self.decoder = Decoder(dec_input, hidden_size, num_layers)

# --- LOAD MODEL & SCALER ---
with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

model = ErosionSeq2Seq().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- DATA PREPARATION ---
print("Preparing test data...")
df = pd.read_csv(TEST_DATA, parse_dates=['dates'])
id_vars = ['dates', 'site_name', 'wave_height', 'wave_period', 'wind_speed', 'tide_level']
transect_cols = [c for c in df.columns if c not in id_vars]
df = df.melt(id_vars=id_vars, value_vars=transect_cols, var_name='transect_id', value_name='distance')
df['unique_id'] = df['site_name'] + "_" + df['transect_id']

# Same robust cleaning as training to ensure consistency
def remove_outliers(group):
    group = group.copy()
    global_median = group['distance'].median()
    is_extreme    = (group['distance'] - global_median).abs() > 150
    group.loc[is_extreme, 'distance'] = np.nan
    rolling_median = group['distance'].rolling(window=5, center=True, min_periods=1).median()
    is_local       = (group['distance'] - rolling_median).abs() > 35
    group.loc[is_local, 'distance'] = np.nan
    group['distance'] = group['distance'].interpolate(method='linear').bfill().ffill()
    return group

df = df.dropna(subset=['distance', 'wave_height', 'wind_speed', 'wave_period', 'tide_level'])
df = df.sort_values(['unique_id', 'dates'])
df = df.groupby('unique_id', group_keys=False).apply(remove_outliers)

# Load test IDs
test_ids  = np.load('data/test_ids_seq2seq.npy', allow_pickle=True)
target_id = test_ids[TRANSECT_IDX]
subset    = df[df['unique_id'] == target_id].sort_values('dates').reset_index(drop=True)

# Generate temporal features
subset['month_sin'] = np.sin(2 * np.pi * subset['dates'].dt.month / 12)
subset['month_cos'] = np.cos(2 * np.pi * subset['dates'].dt.month / 12)

feature_cols_scaled = ['distance', 'wave_height', 'wind_speed', 'wave_period', 'tide_level']
feature_cols_all    = feature_cols_scaled + ['month_sin', 'month_cos'] # 7 features

subset_scaled = subset.copy()
subset_scaled[feature_cols_scaled] = scaler.transform(subset_scaled[feature_cols_scaled])

# Window slicing
start_idx = (subset['dates'] - pd.to_datetime(START_DATE)).abs().idxmin()
all_features = subset_scaled[feature_cols_all].values
enc_block    = all_features[start_idx : start_idx + SEQ_LENGTH]
future_block = all_features[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]

dates_history = subset['dates'].iloc[start_idx : start_idx + SEQ_LENGTH]
dates_future  = subset['dates'].iloc[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]

# --- INVERSE SCALING FUNCTION (FIXED FOR 7 FEATURES) ---
def inverse_distance(scaled_dist, weather_block):
    # The scaler was fit on 5 features. We create a dummy 5-column matrix.
    matrix = np.zeros((len(scaled_dist), 5))
    matrix[:, 0] = scaled_dist
    # We take the first 4 columns of the weather block (wave_h, wind_s, wave_p, tide)
    matrix[:, 1:5] = weather_block[:, :4] 
    return scaler.inverse_transform(matrix)[:, 0]

history_meters       = inverse_distance(enc_block[:, 0], enc_block[:, 1:])
real_observed_meters = inverse_distance(future_block[:, 0], future_block[:, 1:])
base_weather         = future_block[:, 1:].copy() # The 6 weather/time features

# --- AUTOREGRESSIVE PREDICTION ---
def predict_scenario(weather_forcing):
    x_enc = torch.tensor(enc_block, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        _, (h, c) = model.encoder.lstm(x_enc)
        h1, c1 = h[0], c[0]
        h2, c2 = h[1], c[1]

        current_pos = enc_block[-1, 0]  # Start from last known scaled distance
        preds = []

        for t in range(PREDICT_STEPS):
            # Combine current position with the 6 weather/time features for step t
            step_features = [current_pos] + list(weather_forcing[t])
            dec_input = torch.tensor([step_features], dtype=torch.float32).to(device)
            
            pred, h1, c1, h2, c2 = model.decoder.forward_step(dec_input, h1, c1, h2, c2)
            current_pos = pred.item()
            preds.append(current_pos)

    return inverse_distance(np.array(preds), weather_forcing)

# --- RUN SCENARIOS ---
meters_normal  = predict_scenario(base_weather)

weather_extreme = base_weather.copy()
weather_extreme[:, 0] = np.clip(weather_extreme[:, 0] * 1.50, 0, 1) # Wave height +50%
meters_extreme = predict_scenario(weather_extreme)

weather_calm = base_weather.copy()
weather_calm[:, 0] = weather_calm[:, 0] * 0.50 # Wave height -50%
meters_calm = predict_scenario(weather_calm)

# --- RESULTS & PLOT ---
rmse = np.sqrt(np.mean((meters_normal - real_observed_meters) ** 2))
mae  = np.mean(np.abs(meters_normal - real_observed_meters))

print("\n" + "=" * 55)
print(f"TEST TRANSECT RESULTS : {target_id}")
print("=" * 55)
print(f"RMSE: {rmse:.2f} m  |  MAE: {mae:.2f} m")
print("=" * 55)

plt.figure(figsize=(14, 7))
plt.plot(dates_history, history_meters, label='History (CoastSat)', color='black', marker='o', linewidth=2)
plt.plot(dates_future, real_observed_meters, label='True Future (CoastSat)', color='purple', alpha=0.9, linewidth=2.5, marker='s')
plt.plot(dates_future, meters_normal, label='AI Prediction (Real Weather)', color='blue', linestyle='--', linewidth=2)
plt.plot(dates_future, meters_extreme, label='AI Scenario (Storms +50%)', color='red', linestyle=':', linewidth=2)
plt.plot(dates_future, meters_calm, label='AI Scenario (Waves -50%)', color='green', linestyle=':', linewidth=2)

plt.title(f'36-Month Seq2Seq Forecast — Transect {target_id}')
plt.xlabel('Date')
plt.ylabel('Shoreline Position (meters)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('resultat_test_seq2seq.png', dpi=300, bbox_inches='tight')
plt.show()