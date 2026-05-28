import torch
import torch.nn as nn
import numpy as np
import os
import time
import random
from datetime import datetime
from torch.utils.data import DataLoader, TensorDataset

# --- SEQ2SEQ ARCHITECTURE (7 FEATURES) ---

class Encoder(nn.Module):
    def __init__(self, input_size=7, hidden_size=128, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=0.2)
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

    def forward(self, x_enc, x_dec, epsilon=1.0):
        predict_steps = x_dec.size(1)

        _, (h, c) = self.encoder.lstm(x_enc)
        h1, c1 = h[0], c[0]
        h2, c2 = h[1], c[1]

        outputs   = []
        dec_input = x_dec[:, 0, :]  # (batch, 7) — first step

        for t in range(predict_steps):
            pred, h1, c1, h2, c2 = self.decoder.forward_step(dec_input, h1, c1, h2, c2)
            outputs.append(pred)  

            if t < predict_steps - 1:
                # Scheduled sampling
                if random.random() < epsilon:
                    next_pos = x_dec[:, t + 1, 0:1]    # True value
                else:
                    next_pos = pred.detach()           # Model prediction

                next_weather = x_dec[:, t + 1, 1:]     # Remaining 6 features
                dec_input    = torch.cat([next_pos, next_weather], dim=1)

        return torch.stack(outputs, dim=1)  # (batch, 36, 1)


if __name__ == "__main__":
    device     = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    start_time = datetime.now()

    print("=" * 60)
    print(f"START TIME     : {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"DEVICE         : {device.type.upper()}")
    print(f"ARCHITECTURE   : Seq2Seq v3 (7 features) + Scheduled Sampling")
    print("=" * 60)

    model_path = 'model/erosion_seq2seq_v3.pth'
    os.makedirs('model', exist_ok=True)

    print("Loading train data...")
    X_enc = np.load('data/X_enc_train.npy')
    X_dec = np.load('data/X_dec_train.npy')
    y     = np.load('data/y_train.npy')

    dataset    = TensorDataset(torch.from_numpy(X_enc).float(), torch.from_numpy(X_dec).float(), torch.from_numpy(y).float())
    dataloader = DataLoader(dataset, batch_size=512, shuffle=True, num_workers=0)

    model = ErosionSeq2Seq().to(device)

    criterion  = nn.MSELoss()
    optimizer  = torch.optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 100
    scheduler  = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-5)

    print("\nTraining started...")
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        t0 = time.time()
        
        epsilon = max(0.01, 0.95 ** epoch) # Exponential decay for autonomy

        for batch_enc, batch_dec, batch_y in dataloader:
            batch_enc, batch_dec, batch_y = batch_enc.to(device), batch_dec.to(device), batch_y.to(device)

            preds = model(batch_enc, batch_dec, epsilon=epsilon)
            loss  = criterion(preds, batch_y)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(dataloader)
        current_lr = optimizer.param_groups[0]['lr']
        scheduler.step()

        print(f"Epoch [{epoch+1:03d}/{num_epochs}] | Loss: {avg_loss:.6f} | eps: {epsilon:.2f} | lr: {current_lr:.5f}")

    torch.save(model.state_dict(), model_path)
    
    print("\nEvaluating on unseen test sequences (pure autonomy)...")
    X_enc_test = np.load('data/X_enc_test.npy')
    X_dec_test = np.load('data/X_dec_test.npy')
    y_test     = np.load('data/y_test.npy')

    test_dataset = TensorDataset(torch.from_numpy(X_enc_test).float(), torch.from_numpy(X_dec_test).float(), torch.from_numpy(y_test).float())
    test_loader = DataLoader(test_dataset, batch_size=512, shuffle=False)

    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch_enc, batch_dec, batch_y in test_loader:
            batch_enc, batch_dec, batch_y = batch_enc.to(device), batch_dec.to(device), batch_y.to(device)
            preds = model(batch_enc, batch_dec, epsilon=0.0)  # epsilon=0 means 100% autonomy
            test_loss += criterion(preds, batch_y).item()

    print("=" * 60)
    print(f"Train loss : {avg_loss:.6f}")
    print(f"Test loss  : {test_loss/len(test_loader):.6f}")
    print("=" * 60)