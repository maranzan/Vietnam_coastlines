import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import os

# --- 1. CONFIGURATION ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'model/erosion_lstm_multi_v1.pth'
TEST_DATA = 'data/dataset_ready_for_ia.csv'  # Aligned multivariate dataset
SCALER_PATH = 'data/scaler_multi.pkl'
SEQ_LENGTH = 10
PREDICT_STEPS = 36  # Projecting 3 years into the future (36 months)

# --- 2. MULTIVARIATE ARCHITECTURE ---
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

# Load Scaler and Model Weights
if not os.path.exists(SCALER_PATH) or not os.path.exists(MODEL_PATH):
    print("Error: Missing multi-scaler or model weights file.")
    exit()

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

model = ErosionLSTM(input_size=4).to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- 3. TARGET DATA SELECTION ---
df_test = pd.read_csv(TEST_DATA, parse_dates=['dates'])

all_transects = df_test['unique_id'].unique()
target_id = all_transects[20]  
subset = df_test[df_test['unique_id'] == target_id].sort_values('dates').reset_index(drop=True)

# Define your desired start date here (e.g., '2019-01-15')
# It will look for the closest available satellite observation date in your dataset
START_DATE = '2020-06-15'  

# Find the closest index matching that date
start_idx = (subset['dates'] - pd.to_datetime(START_DATE)).abs().idxmin()

# Check if there is enough data left ahead of this date
if start_idx + SEQ_LENGTH + PREDICT_STEPS > len(subset):
    print(f"Error: Not enough data points following {START_DATE} for the simulation window.")
    exit()

# Extract the correctly shifted feature matrices
feature_cols = ['distance', 'wave_height', 'wind_speed', 'wave_period']
all_features = subset[feature_cols].values

# Shift the sequence arrays using our new start index position
history_window = all_features[start_idx : start_idx + SEQ_LENGTH].tolist()
real_observed_scaled = all_features[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS, 0]

dates_history = subset['dates'].iloc[start_idx : start_idx + SEQ_LENGTH]
dates_real_observed = subset['dates'].iloc[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]

print(f"Simulation successfully anchored! History input ends on: {dates_history.iloc[-1].strftime('%Y-%m-%d')}")
print(f"Prospective simulation starts on  : {dates_real_observed.iloc[0].strftime('%Y-%m-%d')}")

# --- 4. SCENARIO GENERATION ENGINE ---
# Extract the true baseline weather forcing associated with the validation window
base_weather = all_features[SEQ_LENGTH : SEQ_LENGTH + PREDICT_STEPS, 1:].copy()

def run_simulation_recursive(weather_forcing):
    """Executes a recursive forecasting loop (open-loop) using a predefined weather matrix."""
    current_window = history_window.copy()
    preds = []
    
    for i in range(PREDICT_STEPS):
        x_input = torch.tensor(current_window[-SEQ_LENGTH:], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_dist = model(x_input).item()
            preds.append(pred_dist)
            
            # Inject the scenario-specific weather variables for the next step
            w_next = weather_forcing[i]
            current_window.append([pred_dist, w_next[0], w_next[1], w_next[2]])
            
    return np.array(preds)

def run_simulation_one_step():
    """Executes a One-Step-Ahead simulation using TRUE past steps (closed-loop verification)."""
    preds = []
    for i in range(PREDICT_STEPS):
        start_idx = i
        end_idx = SEQ_LENGTH + i
        # Passes the true multivariate 10-point block window at each single month step
        x_input = torch.tensor(all_features[start_idx:end_idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            pred_dist = model(x_input).item()
            preds.append(pred_dist)
    return np.array(preds)

# SCENARIO 1: Baseline / Business-As-Usual (Matches the true weather of that period)
weather_normal = base_weather.copy()

# SCENARIO 2: Worst-Case / Typhoon Stress Test (+50% Wave height, +40% Wind speed)
weather_extreme = base_weather.copy()
weather_extreme[:, 0] = np.clip(weather_extreme[:, 0] * 1.50, 0, 1)
weather_extreme[:, 1] = np.clip(weather_extreme[:, 1] * 1.40, 0, 1)

# SCENARIO 3: Protected / Calm Climate (-50% Wave energy, -40% Wind speed)
weather_calm = base_weather.copy()
weather_calm[:, 0] = weather_calm[:, 0] * 0.50
weather_calm[:, 1] = weather_calm[:, 1] * 0.60

print("Running prospective and validation scenario simulations...")
preds_normal = run_simulation_recursive(weather_normal)
preds_extreme = run_simulation_recursive(weather_extreme)
preds_calm = run_simulation_recursive(weather_calm)
preds_one_step = run_simulation_one_step()

# --- 5. INVERSE SCALING WITH MULTIVARIATE CONTEXT ---
def reconstruct_and_inverse(preds_array, weather_forcing):
    matrix = np.zeros((len(preds_array), 4))
    matrix[:, 0] = preds_array
    matrix[:, 1:] = weather_forcing
    return scaler.inverse_transform(matrix)[:, 0]

history_meters = scaler.inverse_transform(np.array(history_window))[:, 0]
real_observed_meters = scaler.inverse_transform(all_features[SEQ_LENGTH : SEQ_LENGTH + PREDICT_STEPS])[:, 0]

meters_normal = reconstruct_and_inverse(preds_normal, weather_normal)
meters_extreme = reconstruct_and_inverse(preds_extreme, weather_extreme)
meters_calm = reconstruct_and_inverse(preds_calm, weather_calm)
meters_one_step = reconstruct_and_inverse(preds_one_step, base_weather)

# --- 6. VISUALIZATION ---
plt.figure(figsize=(14, 7))

# 1. Past Input History
plt.plot(dates_history, history_meters, label='Observed History (10 Input Pts)', color='black', marker='o', linewidth=2)

# 2. True Observed Future Ground Truth
plt.plot(dates_real_observed, real_observed_meters, label='True CoastSat Reality (Observed)', color='purple', alpha=0.9, linewidth=2.5, marker='s')

# 3. AI Validation (One-Step Ahead) - This prevents error accumulation tracking
plt.plot(dates_real_observed, meters_one_step, label='AI Validation: One-Step Ahead (True Past Provided)', color='orange', linestyle='-', linewidth=2.5)

# 4. AI Prospective Recursive Profiles
plt.plot(dates_real_observed, meters_normal, label='AI Scenario: Normal Weather (Recursive)', color='blue', linestyle='--')
plt.plot(dates_real_observed, meters_extreme, label='AI Scenario: Typhoon Stress Test', color='red', linestyle='--')
plt.plot(dates_real_observed, meters_calm, label='AI Scenario: Calm Sea Conditions', color='green', linestyle=':')

plt.title(f'Prospective Scenarios & Validation vs True CoastSat Reality — Transect: {target_id}')
plt.xlabel('Timeline (Dates)')
plt.ylabel('Shoreline Position (Meters)')
plt.legend(loc='best')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Print metrics
rmse_one_step = np.sqrt(np.mean((meters_one_step - real_observed_meters)**2))
print("\n" + "="*50)
print("PROJECTION ENGINE VALIDATION RESULTS")
print("="*50)
print(f"Initial Starting Position       : {history_meters[-1]:.2f} m")
print(f"True Observed CoastSat Reality  : {real_observed_meters[-1]:.2f} m")
print(f"AI Validation (One-Step Ahead)  : {meters_one_step[-1]:.2f} m (RMSE: {rmse_one_step:.2f} m)")
print(f"AI Baseline Projection (Normal) : {meters_normal[-1]:.2f} m")
print(f"AI Typhoon Projection (Extreme) : {meters_extreme[-1]:.2f} m")
print(f"AI Calm Sea Projection (Calm)   : {meters_calm[-1]:.2f} m")
print("="*50)