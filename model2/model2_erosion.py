import torch
import torch.nn as nn
import numpy as np
import os
import time
import platform
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# --- 1. MULTIVARIATE ARCHITECTURE ---
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


if __name__ == "__main__":
    # --- 2. HARDWARE & INITIALIZATION ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = datetime.now()
    processor = platform.processor() or "Unknown Processor"
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"

    print("=" * 60)
    print(f"START TIME      : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"COMPUTE DEVICE  : {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU MODEL       : {gpu_name}")
    else:
        print(f"PROCESSOR       : {processor}")
    print(f"TARGET TYPE     : DELTA (changement de position)")
    print("=" * 60)

    # --- 3. MODEL PATH ---
    model_path = 'model/erosion_lstm_delta_v1.pth'
    os.makedirs('model', exist_ok=True)

    # --- 4. DATA LOADING ---
    if not os.path.exists('data/X_train_delta.npy') or not os.path.exists('data/y_train_delta.npy'):
        print("Error: Run create_sequence_delta.py first.")
        exit()

    print("Loading data arrays...")
    X = np.load('data/X_train_delta.npy')
    y = np.load('data/y_train_delta.npy')

    print(f"Dataset summary : {X.shape[0]} samples | {X.shape[1]} timesteps | {X.shape[2]} features")
    print(f"Delta stats     : mean={y.mean():.6f} | std={y.std():.6f} | min={y.min():.6f} | max={y.max():.6f}")

    dataloader = DataLoader(
        TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).float()),
        batch_size=1024,
        shuffle=True
    )

    # --- 5. INITIALIZE MODEL ---
    model = ErosionLSTM(input_size=4).to(device)

    if os.path.exists(model_path):
        print("Loading existing delta model weights...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except:
            print("Weight mismatch. Starting fresh.")
    else:
        print("No pre-existing weights. Training from scratch.")

    # --- 6. TRAINING ---
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    num_epochs = 30

    print("\nTraining started...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        epoch_start = time.time()

        for batch_X, batch_y in dataloader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        duration = (time.time() - epoch_start) / 60
        avg_loss = epoch_loss / len(dataloader)
        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1:02d}/{num_epochs}] | "
              f"Loss: {avg_loss:.8f} | Duration: {duration:.2f} min")

    # --- 7. SAVE ---
    torch.save(model.state_dict(), model_path)
    total = datetime.now() - start_time

    print("=" * 60)
    print(f"TRAINING COMPLETE AT : {datetime.now().strftime('%H:%M:%S')}")
    print(f"TOTAL EXECUTION TIME : {total}")
    print(f"Model saved          : {model_path}")
    print("=" * 60)