import torch
import torch.nn as nn
import numpy as np
import os
import time
import platform
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# --- ARCHITECTURE SEQ2SEQ ---

class Encoder(nn.Module):
    """Lit les 10 mois d'historique (distance + météo) et produit un contexte."""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)

    def forward(self, x):
        # x : (batch, 10, 4)
        _, (h, c) = self.lstm(x)
        return h, c  # état caché → passé au decoder


class Decoder(nn.Module):
    """Prédit les 36 mois futurs en recevant la météo future à chaque pas."""
    def __init__(self, input_size=4, hidden_size=128, num_layers=2, output_size=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
        self.fc   = nn.Linear(hidden_size, output_size)

    def forward(self, weather_future, h, c):
        # weather_future : (batch, 36, 3)  — météo connue à l'avance
        out, _ = self.lstm(weather_future, (h, c))
        # out : (batch, 36, hidden)
        return self.fc(out)  # → (batch, 36, 1)


class ErosionSeq2Seq(nn.Module):
    def __init__(self, enc_input=4, dec_input=4, hidden_size=128, num_layers=2):
        super().__init__()
        self.encoder = Encoder(enc_input, hidden_size, num_layers)
        self.decoder = Decoder(dec_input, hidden_size, num_layers)

    def forward(self, x_enc, x_dec):
        h, c = self.encoder(x_enc)
        return self.decoder(x_dec, h, c)


if __name__ == "__main__":
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = datetime.now()
    gpu_name   = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU"

    print("=" * 60)
    print(f"START TIME     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DEVICE         : {device.type.upper()}")
    if device.type == 'cuda':
        print(f"GPU            : {gpu_name}")
    print(f"ARCHITECTURE   : Seq2Seq (Encoder 10pts → Decoder 36pts)")
    print("=" * 60)

    model_path = 'model/erosion_seq2seq_v1.pth'
    os.makedirs('model', exist_ok=True)

    # Chargement des données
    for f in ['data/X_enc_seq2seq.npy', 'data/X_dec_seq2seq.npy', 'data/y_seq2seq.npy']:
        if not os.path.exists(f):
            print(f"Error: {f} not found. Run create_sequence_seq2seq.py first.")
            exit()

    print("Loading data...")
    X_enc = np.load('data/X_enc_seq2seq.npy')
    X_dec = np.load('data/X_dec_seq2seq.npy')
    y     = np.load('data/y_seq2seq.npy')

    print(f"Encoder input  : {X_enc.shape}")
    print(f"Decoder input  : {X_dec.shape}")
    print(f"Target         : {y.shape}")

    dataset    = TensorDataset(
        torch.from_numpy(X_enc).float(),
        torch.from_numpy(X_dec).float(),
        torch.from_numpy(y).float()
    )
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    model = ErosionSeq2Seq().to(device)

    if os.path.exists(model_path):
        print("Loading existing weights...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
        except:
            print("Weight mismatch — starting fresh.")
    else:
        print("Training from scratch.")

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5, verbose=True)    
    num_epochs = 30

    print("\nTraining started...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()

        for batch_enc, batch_dec, batch_y in dataloader:
            batch_enc = batch_enc.to(device)
            batch_dec = batch_dec.to(device)
            batch_y   = batch_y.to(device)

            preds = model(batch_enc, batch_dec)   # (batch, 36, 1)
            loss  = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        duration = (time.time() - t0) / 60
        scheduler.step(avg_loss)

        print(f"[{datetime.now().strftime('%H:%M:%S')}] Epoch [{epoch+1:02d}/{num_epochs}] | "
              f"Loss: {avg_loss:.6f} | {duration:.2f} min")

    torch.save(model.state_dict(), model_path)
    total = datetime.now() - start_time

    print("=" * 60)
    print(f"DONE : {datetime.now().strftime('%H:%M:%S')} | Total: {total}")
    print(f"Saved : {model_path}")
    print("=" * 60)
