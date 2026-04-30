import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import os

# --- 1. DEFINE THE ARCHITECTURE (Must match the training script) ---
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

# --- 2. SETUP & LOAD ---
device = torch.device('cpu')
model = ErosionLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)

# Load the saved weights
model_path = 'model/erosion_lstm_v1.pth'
if not os.path.exists(model_path):
    print(f"Error: Model file {model_path} not found!")
    exit()

model.load_state_dict(torch.load(model_path, map_location=device))
model.eval()

# Load the scaler (to convert 0-1 back to meters)
with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Load the dataset to get a sample transect
df = pd.read_csv('data/dataset_ready_for_ia.csv')

# --- 3. RUN PREDICTION ---
# Pick the first unique transect available
target_id = df['unique_id'].unique()[0] 
data_series = df[df['unique_id'] == target_id]['distance_scaled'].values

# Prepare input: Take the last 10 known points
# Shape needed for LSTM: [Batch_size=1, Time_steps=10, Features=1]
input_seq = torch.from_numpy(data_series[-10:]).float().view(1, 10, 1)

with torch.no_grad():
    pred_scaled = model(input_seq)
    # Convert prediction and actual history back to meters
    prediction_meters = scaler.inverse_transform(pred_scaled.numpy())
    history_meters = scaler.inverse_transform(data_series[-20:].reshape(-1, 1))

# --- 4. VISUALIZATION ---
plt.figure(figsize=(12, 6))
plt.plot(range(20), history_meters, label="Historical Data (Last 20 dates)", marker='o', color='#1f77b4')

plt.scatter(20, prediction_meters, color='red', s=150, label="AI Prediction (Future)", edgecolors='black', zorder=5)

plt.title(f"Shoreline Prediction Analysis: {target_id}", fontsize=14)
plt.ylabel("Cross-shore Distance (meters)", fontsize=12)
plt.xlabel("Relative Time Steps", fontsize=12)
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)

# Show the plotimport torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import os



if __name__ == "__main__":
        
    # --- 1. DATA LOADING ---
    if not os.path.exists('data/X_train.npy'):
        print("Error: Sequences not found. Please run prepare_sequences.py first.")
        exit()

    X = np.load('data/X_train.npy')
    y = np.load('data/y_train.npy')


    X_tensor = torch.from_numpy(X).float()
    y_tensor = torch.from_numpy(y).float()


    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    # --- 2. ARCHITECTURE DEFINITION ---
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

    # --- 3. INSTANTIATION ---
    model = ErosionLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # --- 4. TRAINING SETUP ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # --- 5. TRAINING LOOP ---
    num_epochs = 100
    print(f"Training started on {device}...")

    if os.path.exists('model/erosion_lstm_v1.pth'):
        model.load_state_dict(torch.load('model/erosion_lstm_v1.pth'))
        print("reusing existing weights to continue training...")

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
        
        if (epoch+1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.6f}')

    print("Training complete!")

    # --- 6. SAVE THE MODEL ---
    os.makedirs('model', exist_ok=True)
    torch.save(model.state_dict(), 'model/erosion_lstm_v1.pth')
    print("Model saved successfully in 'model/erosion_lstm_v1.pth'")



plt.show()

print(f"Analysis for {target_id}:")
print(f"   - Current position: {history_meters[-1][0]:.2f} m")
print(f"   - Predicted next position: {prediction_meters[0][0]:.2f} m")