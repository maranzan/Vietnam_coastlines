import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os
from torch.utils.data import DataLoader, TensorDataset

# --- 1. ARCHITECTURE ---
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

if __name__ == "__main__":
    # --- 2. INITIALISATION ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = ErosionLSTM().to(device)
    model_path = 'model/erosion_lstm_v1.pth'
    os.makedirs('model', exist_ok=True)

    # --- 3. CHARGEMENT DES DONNÉES D'ENTRAÎNEMENT ---
    if not os.path.exists('data/X_train.npy'):
        print("Error: Run prepare_sequences.py first.")
        exit()

    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')
    dataloader = DataLoader(TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()), batch_size=32, shuffle=True)

    # --- 4. LOGIQUE DE CHARGEMENT / RESET ---
    if os.path.exists(model_path):
        print("Reusing existing weights...")
        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    else:
        print("No model found. Starting fresh training from scratch!")

    # --- 5. ENTRAÎNEMENT ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20 

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.6f}')

    # Sauvegarde finale
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

    # --- 6. VISUALISATION RAPIDE (OPTIONNEL) ---
    # Ici tu peux remettre ton code de plt.show() si tu veux voir un résultat


    plt.show()

    print(f"Analysis for {target_id}:")
    print(f"   - Current position: {history_meters[-1][0]:.2f} m")
    print(f"   - Predicted next position: {prediction_meters[0][0]:.2f} m")