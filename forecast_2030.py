import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

class ErosionLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
        super(ErosionLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        h0 = torch.zeros(2, x.size(0), 64).to(x.device)
        c0 = torch.zeros(2, x.size(0), 64).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        return self.fc(out[:, -1, :])

# --- LOAD MODEL & SCALER ---
model = ErosionLSTM()
model.load_state_dict(torch.load('model/erosion_lstm_v1.pth'))
model.eval()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df = pd.read_csv('data/dataset_ready_for_ia.csv')
target_id = df['unique_id'].unique()[0] 
data_series = list(df[df['unique_id'] == target_id]['distance_scaled'].values)

# --- FORECAST LOOP (Predict next 24 months) ---
n_months_to_forecast = 24
forecast_results = []

current_sequence = data_series[-10:]

for _ in range(n_months_to_forecast):
    input_tensor = torch.tensor(current_sequence[-10:]).float().view(1, 10, 1)
    with torch.no_grad():
        pred = model(input_tensor).item()
        forecast_results.append(pred)
        current_sequence.append(pred)

# --- VISUALIZATION ---
history_m = scaler.inverse_transform(np.array(data_series[-30:]).reshape(-1, 1))
forecast_m = scaler.inverse_transform(np.array(forecast_results).reshape(-1, 1))

plt.figure(figsize=(12, 5))
plt.plot(range(30), history_m, label="Historical Data")
plt.plot(range(30, 30 + n_months_to_forecast), forecast_m, label="AI Forecast", linestyle='--', color='red')
plt.axvline(x=30, color='black', linestyle=':', label="Today")
plt.title(f"Future Shoreline Forecast for {target_id} (2 years)")
plt.ylabel("Distance (m)")
plt.legend()
plt.show()