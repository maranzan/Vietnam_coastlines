import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- 1. CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'model/erosion_lstm_delta_v1.pth'
TEST_DATA = 'data/dataset_ready_for_ia.csv'
SCALER_PATH = 'data/scaler_delta.pkl'
DELTA_SCALER_PATH = 'data/delta_scaler.pkl'
SEQ_LENGTH = 10
PREDICT_STEPS = 36

# --- 2. ARCHITECTURE ---
class ErosionLSTM(nn.Module):
    def __init__(self, input_size=4, hidden_size=64, num_layers=2, output_size=1):
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

# --- 3. CHARGEMENT MODÈLE & SCALERS ---
for path in [SCALER_PATH, DELTA_SCALER_PATH, MODEL_PATH]:
    if not os.path.exists(path):
        print(f"Error: Missing {path}")
        exit()

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

with open(DELTA_SCALER_PATH, 'rb') as f:
    delta_scaler = pickle.load(f)

model = ErosionLSTM(input_size=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- 4. SÉLECTION DU TRANSECT ---
df_test = pd.read_csv(TEST_DATA, parse_dates=['dates'])
all_transects = df_test['unique_id'].unique()
target_id = all_transects[200]
subset = df_test[df_test['unique_id'] == target_id].sort_values('dates').reset_index(drop=True)

START_DATE = '2018-06-15'
start_idx = (subset['dates'] - pd.to_datetime(START_DATE)).abs().idxmin()

if start_idx + SEQ_LENGTH + PREDICT_STEPS > len(subset):
    print(f"Error: Not enough data after {START_DATE}.")
    exit()

feature_cols = ['distance', 'wave_height', 'wind_speed', 'wave_period']
all_features = subset[feature_cols].values

history_window = all_features[start_idx : start_idx + SEQ_LENGTH].tolist()
dates_history = subset['dates'].iloc[start_idx : start_idx + SEQ_LENGTH]
dates_future  = subset['dates'].iloc[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]

print(f"History ends on  : {dates_history.iloc[-1].strftime('%Y-%m-%d')}")
print(f"Forecast starts  : {dates_future.iloc[0].strftime('%Y-%m-%d')}")

# --- 5. HELPER : INVERSE SCALING ---
def inverse_distance(scaled_values, weather_block):
    """Reconstruit les mètres réels depuis les valeurs scalées de distance."""
    matrix = np.zeros((len(scaled_values), 4))
    matrix[:, 0] = scaled_values
    matrix[:, 1:] = weather_block
    return scaler.inverse_transform(matrix)[:, 0]

# Position historique et vérité terrain
history_meters = inverse_distance(
    np.array(history_window)[:, 0],
    np.array(history_window)[:, 1:]
)

real_block = all_features[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]
real_observed_meters = inverse_distance(real_block[:, 0], real_block[:, 1:])

base_weather = all_features[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS, 1:].copy()

# --- 6. SIMULATION RÉCURSIVE ---
def run_simulation_recursive(weather_forcing):
    """
    Le modèle prédit un delta normalisé (StandardScaler).
    On inverse-transforme chaque delta avant de l'accumuler → pas de drift.
    """
    current_window = history_window.copy()
    predicted_pos_scaled = []

    for i in range(PREDICT_STEPS):
        x_input = torch.tensor(
            current_window[-SEQ_LENGTH:], dtype=torch.float32
        ).unsqueeze(0).to(device)

        with torch.no_grad():
            pred_delta_normalized = model(x_input).item()

        # Inverse-transforme le delta : espace normalisé → espace scalé MinMax
        pred_delta_minmax = delta_scaler.inverse_transform([[pred_delta_normalized]])[0][0]

        # Accumule dans l'espace scalé MinMax
        new_pos_scaled = current_window[-1][0] + pred_delta_minmax

        # Injection dans la fenêtre pour le prochain pas
        w_next = weather_forcing[i]
        current_window.append([new_pos_scaled, w_next[0], w_next[1], w_next[2]])
        predicted_pos_scaled.append(new_pos_scaled)

    # Inverse-transforme toutes les positions vers les mètres réels
    return inverse_distance(np.array(predicted_pos_scaled), weather_forcing)


# --- 7. VALIDATION ONE-STEP-AHEAD ---
def run_simulation_one_step():
    """Utilise les vraies valeurs passées à chaque pas (fermé) — sert de référence."""
    preds_scaled = []
    for i in range(PREDICT_STEPS):
        s = start_idx + i
        e = start_idx + SEQ_LENGTH + i
        x_input = torch.tensor(
            all_features[s:e], dtype=torch.float32
        ).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_delta_normalized = model(x_input).item()

        # Inverse-transforme le delta
        pred_delta_minmax = delta_scaler.inverse_transform([[pred_delta_normalized]])[0][0]

        # Position vraie à t + delta prédit
        true_pos_at_t = all_features[e - 1, 0]
        predicted_pos_scaled = true_pos_at_t + pred_delta_minmax
        preds_scaled.append(predicted_pos_scaled)

    return inverse_distance(np.array(preds_scaled), base_weather)


# --- 8. SCÉNARIOS MÉTÉO ---
weather_normal  = base_weather.copy()

weather_extreme = base_weather.copy()
weather_extreme[:, 0] = np.clip(weather_extreme[:, 0] * 1.50, 0, 1)
weather_extreme[:, 1] = np.clip(weather_extreme[:, 1] * 1.40, 0, 1)

weather_calm = base_weather.copy()
weather_calm[:, 0] = weather_calm[:, 0] * 0.50
weather_calm[:, 1] = weather_calm[:, 1] * 0.60

print("Running simulations...")
meters_normal   = run_simulation_recursive(weather_normal)
meters_extreme  = run_simulation_recursive(weather_extreme)
meters_calm     = run_simulation_recursive(weather_calm)
meters_one_step = run_simulation_one_step()

# --- 9. VISUALISATION ---
plt.figure(figsize=(14, 7))

plt.plot(dates_history, history_meters,
         label='Observed History (10 Input Pts)', color='black', marker='o', linewidth=2)
plt.plot(dates_future, real_observed_meters,
         label='True CoastSat Reality', color='purple', alpha=0.9, linewidth=2.5, marker='s')
plt.plot(dates_future, meters_one_step,
         label='AI Validation: One-Step Ahead', color='orange', linewidth=2.5)
plt.plot(dates_future, meters_normal,
         label='AI Scenario: Normal Weather (Recursive)', color='blue', linestyle='--')
plt.plot(dates_future, meters_extreme,
         label='AI Scenario: Typhoon Stress Test', color='red', linestyle='--')
plt.plot(dates_future, meters_calm,
         label='AI Scenario: Calm Sea Conditions', color='green', linestyle=':')

plt.title(f'Prospective Scenarios & Validation vs True CoastSat Reality — {target_id}')
plt.xlabel('Timeline (Dates)')
plt.ylabel('Shoreline Position (Meters)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# --- 10. MÉTRIQUES ---
rmse_one_step = np.sqrt(np.mean((meters_one_step - real_observed_meters) ** 2))

print("\n" + "=" * 50)
print("PROJECTION ENGINE — DELTA MODEL RESULTS")
print("=" * 50)
print(f"Starting Position               : {history_meters[-1]:.2f} m")
print(f"True CoastSat Final Position    : {real_observed_meters[-1]:.2f} m")
print(f"AI One-Step Final Position      : {meters_one_step[-1]:.2f} m  (RMSE: {rmse_one_step:.2f} m)")
print(f"AI Normal Scenario Final        : {meters_normal[-1]:.2f} m")
print(f"AI Typhoon Scenario Final       : {meters_extreme[-1]:.2f} m")
print(f"AI Calm Scenario Final          : {meters_calm[-1]:.2f} m")
print("=" * 50)