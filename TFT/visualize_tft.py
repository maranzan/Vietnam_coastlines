import os
import warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import (
    GroupNormalizer, TorchNormalizer, EncoderNormalizer
)

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

# Fix PyTorch 2.6+
torch.serialization.add_safe_globals([
    GroupNormalizer, NaNLabelEncoder, TorchNormalizer, EncoderNormalizer,
    np.core.multiarray.scalar, np.dtype, np.float64, np.int64
])

# --- CONFIGURATION ---
SEQ_LENGTH    = 30
PREDICT_STEPS = 36
CHECKPOINT    = 'model/tft_last.ckpt'
TRANSECT_IDX  = 151
PLOT_HISTORY  = 25

# --- LOAD DATA ---
print("Loading dataset...")
df = pd.read_csv('data/dataset_tft.csv')
df['dates'] = pd.to_datetime(df['dates'])
df = df.reset_index(drop=True)

test_ids = np.load('data/test_ids.npy', allow_pickle=True)
if TRANSECT_IDX >= len(test_ids):
    raise IndexError(f"TRANSECT_IDX out of bounds (0 to {len(test_ids)-1}).")

target_id = test_ids[TRANSECT_IDX]
print(f"Test transect : {target_id}")

# --- RECONSTRUCT TRAINING DATASET ---
# Required to properly initialize the GroupNormalizer with the exact same stats
max_time = df.groupby('unique_id')['time_idx'].transform('max')
cutoff   = (max_time * 0.80).astype(int)
df_train = df[df['time_idx'] <= cutoff].reset_index(drop=True)

training = TimeSeriesDataSet(
    df_train,
    time_idx                   = 'time_idx',
    target                     = 'distance',
    group_ids                  = ['unique_id'],
    min_encoder_length         = SEQ_LENGTH // 2,
    max_encoder_length         = SEQ_LENGTH,
    min_prediction_length      = 1,
    max_prediction_length      = PREDICT_STEPS,
    time_varying_known_reals   = [
        'time_idx', 'wave_height', 'wind_speed', 'wave_period', 'tide_level',
        'month_sin', 'month_cos'
    ],
    time_varying_unknown_reals = ['distance'],
    static_categoricals        = ['unique_id', 'site_name'],
    categorical_encoders       = {
        'unique_id': NaNLabelEncoder(add_nan=True),
        'site_name': NaNLabelEncoder(add_nan=True),
    },
    target_normalizer          = 'auto',
    allow_missing_timesteps    = True,
)

# --- LOAD MODEL ---
print(f"Loading checkpoint : {CHECKPOINT}")
tft = TemporalFusionTransformer.load_from_checkpoint(CHECKPOINT)
tft.eval()

# --- SELECT TEST WINDOW ---
subset    = df[df['unique_id'] == target_id].sort_values('time_idx').reset_index(drop=True)
start_idx = len(subset) - (SEQ_LENGTH + PREDICT_STEPS)

if start_idx < 0:
    print(f"Not enough data points for {target_id}. Try another TRANSECT_IDX.")
    exit()

history_block = subset.iloc[start_idx : start_idx + SEQ_LENGTH]
future_block  = subset.iloc[start_idx + SEQ_LENGTH : start_idx + SEQ_LENGTH + PREDICT_STEPS]
dates_history = history_block['dates']
dates_future  = future_block['dates']
real_observed = future_block['distance'].values

# --- PREDICTION ---
def predict_scenario(scenario_df):
    ds = TimeSeriesDataSet.from_dataset(
        training, scenario_df,
        predict            = True,
        stop_randomization = True,
        allow_missing_timesteps = True,
    )
    loader = ds.to_dataloader(train=False, batch_size=1, num_workers=0)
    
    with torch.no_grad():
        raw = tft.predict(loader, mode='raw', return_x=False)

    # Automatically handle MAE (1 output) vs QuantileLoss (multiple outputs)
    raw_preds = raw['prediction'][0] # shape: (time_steps, num_outputs)
    
    if raw_preds.shape[-1] == 1:
        # Model was trained with MAE (Point prediction)
        pred = raw_preds[:, 0].cpu().numpy()
    else:
        # Model was trained with QuantileLoss (Index 3 is usually the median/50th percentile)
        pred = raw_preds[:, 3].cpu().numpy()
        
    return pred

def build_scenario(multipliers: dict):
    s = subset.copy()
    future_mask = s.index >= (start_idx + SEQ_LENGTH)
    for col, mult in multipliers.items():
        s.loc[future_mask, col] = np.clip(s.loc[future_mask, col] * mult, 0, None)
    return s

print("Running predictions...")
pred_n = predict_scenario(build_scenario({}))
pred_e = predict_scenario(build_scenario({'wave_height': 1.50, 'wind_speed': 1.40})) # Storm
pred_c = predict_scenario(build_scenario({'wave_height': 0.50, 'wind_speed': 0.60})) # Calm

rmse = np.sqrt(np.mean((pred_n - real_observed) ** 2))
mae  = np.mean(np.abs(pred_n - real_observed))
print(f"Normal RMSE : {rmse:.2f} m")
print(f"Normal MAE  : {mae:.2f} m")

# --- PLOT RESULTS ---
plot_dates_history = dates_history.iloc[-PLOT_HISTORY:]
plot_dist_history  = history_block['distance'].iloc[-PLOT_HISTORY:]

fig, ax = plt.subplots(figsize=(14, 7))

ax.plot(plot_dates_history, plot_dist_history,
        label=f'Observed History (last {PLOT_HISTORY} pts)', color='black', marker='o', linewidth=2)
ax.plot(dates_future, real_observed,
        label='True CoastSat Reality', color='purple', linewidth=2.5, marker='s', alpha=0.9)
ax.plot(dates_future, pred_n,
        label='TFT: Normal Weather', color='blue', linestyle='--', linewidth=2)
ax.plot(dates_future, pred_e,
        label='TFT: Typhoon Stress Test (+50% waves)', color='red', linestyle='--', linewidth=2)
ax.plot(dates_future, pred_c,
        label='TFT: Calm Sea (-50% waves)', color='green', linestyle=':', linewidth=2)

# Dynamic Y-axis adjustment
all_y = np.concatenate([plot_dist_history.values, real_observed, pred_n, pred_e, pred_c])
y_min, y_max = np.nanmin(all_y), np.nanmax(all_y)
pad = (y_max - y_min) * 0.1
ax.set_ylim(y_min - pad, y_max + pad)

ax.set_title(f'TFT {PREDICT_STEPS}-Month Forecast — {target_id}\n'
             f'RMSE: {rmse:.2f} m | MAE: {mae:.2f} m')
ax.set_xlabel('Timeline (Dates)')
ax.set_ylabel('Shoreline Position (Meters)')
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.4, linestyle='--')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()