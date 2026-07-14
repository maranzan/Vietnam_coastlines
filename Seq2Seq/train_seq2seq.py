import pandas as pd
import numpy as np
import os
import json
import pickle
import random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
SEQ_LENGTH    = 30
PREDICT_STEPS = 36   
TEST_RATIO    = 0.15
VAL_RATIO     = 0.15 

BATCH_SIZE    = 256
MAX_EPOCHS    = 150
HIDDEN_SIZE   = 128
NUM_LAYERS    = 3
DROPOUT       = 0.3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# --- 2. DATA PREPARATION ---
print("Loading dataset...")
df = pd.read_csv('data/vietnam_ml_dataset.csv')
df['date'] = pd.to_datetime(df['date'])

df = df[df['in_river_zone'] == 0].copy()

# --- NOUVEAU : DÉTECTION DYNAMIQUE DES VARIABLES VALIDES ---
feature_cols_base = ['cross_distance_m', 'Hs_mean_7d', 'wind_mean_7d', 'wave_period_s', 'tide_height_m']
valid_cols = []
for col in feature_cols_base:
    nan_ratio = df[col].isna().mean()
    if nan_ratio > 0.5:
        print(f"⚠️ IGNORÉ : '{col}' (Contient {nan_ratio*100:.1f}% de NaNs)")
    else:
        valid_cols.append(col)

feature_cols_base = valid_cols

# Sauvegarde des features utilisées pour l'inférence
os.makedirs('data', exist_ok=True)
with open('data/features_seq2seq.json', 'w') as f:
    json.dump(feature_cols_base, f)

df = df.dropna(subset=feature_cols_base)
if len(df) == 0:
    raise ValueError("Le dataset est complètement vide après avoir retiré les NaNs des variables valides !")

def remove_outliers(group):
    group = group.copy()
    global_median = group['cross_distance_m'].median()
    is_extreme    = (group['cross_distance_m'] - global_median).abs() > 150
    group.loc[is_extreme, 'cross_distance_m'] = np.nan
    
    rolling_median = group['cross_distance_m'].rolling(window=5, center=True, min_periods=1).median()
    is_local       = (group['cross_distance_m'] - rolling_median).abs() > 35
    group.loc[is_local, 'cross_distance_m'] = np.nan
    
    group['cross_distance_m'] = group['cross_distance_m'].interpolate(method='linear').bfill().ffill()
    return group

df = df.sort_values(['segment_id', 'date'])
df = df.groupby('segment_id', group_keys=False).apply(remove_outliers)
df = df.dropna(subset=feature_cols_base)

df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12)

# --- 3. STRICT DATA SPLIT ---
np.random.seed(42)
all_ids = df['segment_id'].unique()
np.random.shuffle(all_ids)

n = len(all_ids)
test_idx = int(TEST_RATIO * n)
val_idx  = test_idx + int(VAL_RATIO * n)

test_ids  = all_ids[:test_idx]
val_ids   = all_ids[test_idx:val_idx]
train_ids = all_ids[val_idx:]

np.save('data/test_ids_seq2seq.npy', test_ids)

df_train = df[df['segment_id'].isin(train_ids)].copy()
df_val   = df[df['segment_id'].isin(val_ids)].copy()
df_test  = df[df['segment_id'].isin(test_ids)].copy()

# --- 4. SCALING ---
feature_cols_scaled = feature_cols_base.copy()
feature_cols_all    = feature_cols_scaled + ['month_sin', 'month_cos']
TOTAL_FEATURES      = len(feature_cols_all)

print("Standardizing (Z-score)...")
scaler = StandardScaler()
df_train[feature_cols_scaled] = scaler.fit_transform(df_train[feature_cols_scaled])
df_val[feature_cols_scaled]   = scaler.transform(df_val[feature_cols_scaled])
df_test[feature_cols_scaled]  = scaler.transform(df_test[feature_cols_scaled])

with open('data/scaler_seq2seq.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# --- 5. SEQUENCE GENERATION ---
def create_sequences(data_df):
    features_array = data_df[feature_cols_all].values.astype(np.float32)
    ids_array      = data_df['segment_id'].values
    total_window   = SEQ_LENGTH + PREDICT_STEPS

    X_windows = np.lib.stride_tricks.sliding_window_view(
        features_array, window_shape=(total_window, features_array.shape[1])
    ).squeeze(1)

    id_encoded = pd.factorize(ids_array)[0]
    id_changes = np.diff(id_encoded) != 0
    change_windows = np.lib.stride_tricks.sliding_window_view(id_changes, window_shape=total_window - 1)
    
    valid_mask = ~change_windows.any(axis=1)
    valid_indices = np.where(valid_mask)[0]
    
    X_enc = X_windows[valid_indices, :SEQ_LENGTH, :]
    chunk_dec = X_windows[valid_indices, SEQ_LENGTH:, :]
    
    y = chunk_dec[:, :, 0:1]
    
    X_dec = np.empty((len(valid_indices), PREDICT_STEPS, TOTAL_FEATURES), dtype=np.float32)
    last_known = np.empty((len(valid_indices), PREDICT_STEPS, 1), dtype=np.float32)
    last_known[:, 0:1, :] = X_enc[:, -1:, 0:1]
    last_known[:, 1:, :]  = chunk_dec[:, :-1, 0:1]
    
    X_dec[:, :, 0:1] = last_known
    X_dec[:, :, 1:]  = chunk_dec[:, :, 1:]
    
    return X_enc, X_dec, y

print("Generating sequences...")
X_enc_train, X_dec_train, y_train = create_sequences(df_train)
X_enc_val, X_dec_val, y_val       = create_sequences(df_val)
X_enc_test, X_dec_test, y_test    = create_sequences(df_test)

train_loader = DataLoader(TensorDataset(torch.from_numpy(X_enc_train), torch.from_numpy(X_dec_train), torch.from_numpy(y_train)), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(TensorDataset(torch.from_numpy(X_enc_val), torch.from_numpy(X_dec_val), torch.from_numpy(y_val)), batch_size=BATCH_SIZE, shuffle=False)
test_loader  = DataLoader(TensorDataset(torch.from_numpy(X_enc_test), torch.from_numpy(X_dec_test), torch.from_numpy(y_test)), batch_size=BATCH_SIZE, shuffle=False)

# --- 6. ARCHITECTURE BI-LSTM DYNAMIQUE ---
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True, dropout=DROPOUT)
    def forward(self, x):
        out, _ = self.lstm(x)
        return out.mean(dim=1)

class Decoder(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=3, output_size=1):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        lstm_input_size = input_size + (2 * hidden_size)
        self.lstm_cells = nn.ModuleList([
            nn.LSTMCell(lstm_input_size if i == 0 else hidden_size, hidden_size) for i in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, output_size)

    def forward_step(self, x, context, h_states, c_states):
        lstm_in = torch.cat([x, context], dim=1)
        new_h, new_c = [], []
        for i, cell in enumerate(self.lstm_cells):
            h, c = cell(lstm_in, (h_states[i], c_states[i]))
            new_h.append(h)
            new_c.append(c)
            lstm_in = h
        return self.fc(new_h[-1]), new_h, new_c

class BiLSTMErosionModel(nn.Module):
    def __init__(self, enc_input, dec_input, hidden_size=128, num_layers=3):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.encoder = Encoder(enc_input, hidden_size, num_layers)
        self.decoder = Decoder(dec_input, hidden_size, num_layers)

    def forward(self, x_enc, x_dec, epsilon=1.0):
        batch_size = x_enc.size(0)
        predict_steps = x_dec.size(1)
        context = self.encoder(x_enc)
        h_states = [torch.zeros(batch_size, self.hidden_size).to(x_enc.device) for _ in range(self.num_layers)]
        c_states = [torch.zeros(batch_size, self.hidden_size).to(x_enc.device) for _ in range(self.num_layers)]
        outputs = []
        dec_input = x_dec[:, 0, :]
        for t in range(predict_steps):
            pred, h_states, c_states = self.decoder.forward_step(dec_input, context, h_states, c_states)
            outputs.append(pred)  
            if t < predict_steps - 1:
                next_pos = x_dec[:, t + 1, 0:1] if random.random() < epsilon else pred.detach()
                next_weather = x_dec[:, t + 1, 1:]
                dec_input = torch.cat([next_pos, next_weather], dim=1)
        return torch.stack(outputs, dim=1)

model = BiLSTMErosionModel(enc_input=TOTAL_FEATURES, dec_input=TOTAL_FEATURES, hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS).to(device)

# --- 7. TRAINING ---
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5) 
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

best_val_loss = float('inf')
patience_counter = 0
EARLY_STOPPING_PATIENCE = 15

os.makedirs('model', exist_ok=True)
best_model_path = 'model/erosion_bilstm_best.pth'

print("\nStarting Training...")
for epoch in range(MAX_EPOCHS):
    model.train()
    train_loss = 0
    epsilon = max(0.0, 1.0 - (epoch / (MAX_EPOCHS * 0.5)))

    for batch_enc, batch_dec, batch_y in train_loader:
        batch_enc, batch_dec, batch_y = batch_enc.to(device), batch_dec.to(device), batch_y.to(device)
        optimizer.zero_grad()
        preds = model(batch_enc, batch_dec, epsilon=epsilon)
        loss = criterion(preds, batch_y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_enc, batch_dec, batch_y in val_loader:
            batch_enc, batch_dec, batch_y = batch_enc.to(device), batch_dec.to(device), batch_y.to(device)
            preds = model(batch_enc, batch_dec, epsilon=0.0)
            val_loss += criterion(preds, batch_y).item()
            
    avg_val_loss = val_loss / len(val_loader)
    scheduler.step(avg_val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    
    print(f"Epoch [{epoch+1:03d}/{MAX_EPOCHS}] | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | eps: {epsilon:.2f} | lr: {current_lr}")

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}!")
            break

# --- 8. EVALUATION ---
print("\nEvaluation on Test Set...")
model.load_state_dict(torch.load(best_model_path))
model.eval()
all_preds, all_trues = [], []
with torch.no_grad():
    for batch_enc, batch_dec, batch_y in test_loader:
        batch_enc, batch_dec, batch_y = batch_enc.to(device), batch_dec.to(device), batch_y.to(device)
        preds = model(batch_enc, batch_dec, epsilon=0.0)
        all_preds.append(preds.cpu().numpy())
        all_trues.append(batch_y.cpu().numpy())

preds_flat = np.concatenate(all_preds).flatten()
trues_flat = np.concatenate(all_trues).flatten()

rmse = np.sqrt(mean_squared_error(trues_flat, preds_flat))
mae = mean_absolute_error(trues_flat, preds_flat)
r2 = r2_score(trues_flat, preds_flat)

print(f"RMSE : {rmse:.4f} | MAE : {mae:.4f} | R² : {r2:.4f}")