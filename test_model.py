import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

# --- 1. PARAMÈTRES ET CHARGEMENT ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = 'model/erosion_lstm_v1.pth'
TEST_DATA = 'data/test_set_region.csv'
SCALER_PATH = 'data/scaler.pkl'
SEQ_LENGTH = 10  # Doit être le même que pendant l'entraînement

with open(SCALER_PATH, 'rb') as f:
    scaler = pickle.load(f)

# --- 2. DÉFINITION DE L'ARCHITECTURE (Doit être identique à ton entraînement) ---
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

# Initialisation et chargement du modèle sur le GPU
model = ErosionLSTM().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device, weights_only=True))
model.eval()

# --- 3. PRÉPARATION DES SÉQUENCES DE TEST ---
df_test = pd.read_csv(TEST_DATA, parse_dates=['dates'])
# On choisit un transect au hasard parmi le jeu de test pour la visualisation
target_id = df_test['unique_id'].iloc[500]
subset = df_test[df_test['unique_id'] == target_id].sort_values('dates')

# Transformation en séquences pour le LSTM
data = subset['distance_scaled'].values
X_test = []
y_real = []

for i in range(len(data) - SEQ_LENGTH):
    X_test.append(data[i:i+SEQ_LENGTH])
    y_real.append(data[i+SEQ_LENGTH])

X_test = torch.tensor(np.array(X_test), dtype=torch.float32).unsqueeze(-1).to(device)

# --- 4. PRÉDICTION ---
with torch.no_grad():
    predictions = model(X_test).cpu().numpy()

# Inverser la normalisation pour repasser en mètres
predictions_meters = scaler.inverse_transform(predictions)
y_real_meters = scaler.inverse_transform(np.array(y_real).reshape(-1, 1))

# --- 5. VISUALISATION ET MÉTRIQUES ---
plt.figure(figsize=(12, 6))
plt.plot(subset['dates'].iloc[SEQ_LENGTH:], y_real_meters, label='Réalité (CoastSat)', color='blue', marker='o')
plt.plot(subset['dates'].iloc[SEQ_LENGTH:], predictions_meters, label='Prédiction IA', color='red', linestyle='--')
plt.title(f'Comparaison Érosion : {target_id}')
plt.xlabel('Date')
plt.ylabel('Distance (m)')
plt.legend()
plt.grid(True)
plt.show()

# Calcul de l'erreur moyenne en mètres
rmse = np.sqrt(np.mean((predictions_meters - y_real_meters)**2))
print(f"--- RÉSULTATS POUR {target_id} ---")
print(f"Erreur moyenne (RMSE) : {rmse:.2f} mètres")