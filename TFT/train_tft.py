import os
import argparse
import warnings
import numpy as np
import pandas as pd
import torch
import torch.serialization
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import QuantileLoss
from pytorch_forecasting.data import NaNLabelEncoder
from pytorch_forecasting.metrics import MAE
from pytorch_forecasting.data.encoders import (
    GroupNormalizer, TorchNormalizer, EncoderNormalizer
)

warnings.filterwarnings('ignore')
torch.set_float32_matmul_precision('high')

# Fix PyTorch 2.6+
torch.serialization.add_safe_globals([
    GroupNormalizer,
    NaNLabelEncoder,
    TorchNormalizer,
    EncoderNormalizer,
    np.core.multiarray.scalar,
    np.dtype,
    np.float64,
    np.int64,
])

# --- IMPROVED CONFIGURATION ---
SEQ_LENGTH    = 30   # Adjusted to 30 to give roughly 1 month of history
PREDICT_STEPS = 36   # Forecast horizon unchanged
BATCH_SIZE    = 128  # Reduced from 256 to 128 to add beneficial stochasticity
MAX_EPOCHS    = 50
LEARNING_RATE = 0.0005 # Slightly lowered for more stable convergence

# --- RESUME ARGUMENT ---
parser = argparse.ArgumentParser()
parser.add_argument('--resume', action='store_true', help='Resume from the last checkpoint')
args = parser.parse_args()

# --- DATA LOADING ---
print("Loading dataset...")
df_all = pd.read_csv('data/dataset_tft.csv')
df_all['dates'] = pd.to_datetime(df_all['dates'])
df_all = df_all.reset_index(drop=True)

# --- STRICT TEMPORAL SPLIT (Solves the val_loss issue) ---
# We ignore the .npy files because we want ALL transects in Train and Val, separated in time.
print("Applying temporal split (80/20) on all transects...")
max_time = df_all.groupby('unique_id')['time_idx'].transform('max')
cutoff   = (max_time * 0.80).astype(int)

df_train = df_all[df_all['time_idx'] <= cutoff].reset_index(drop=True)
df_val   = df_all.reset_index(drop=True)  # Contains the full history so the encoder can work

print(f"Train : {len(df_train):,} observations | {df_train['unique_id'].nunique()} transects")
print(f"Val   : {len(df_val):,} observations   | {df_val['unique_id'].nunique()} transects")

# --- TIMESERIESDATASET ---
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
    static_categoricals        = ['unique_id', 'site_name'], # The model also learns site-specific features
    categorical_encoders       = {
        'unique_id': NaNLabelEncoder(add_nan=True),
        'site_name': NaNLabelEncoder(add_nan=True)
    },
    target_normalizer          = 'auto',  # GroupNormalizer is essential to handle scale differences (100m vs 1000m)
    allow_missing_timesteps    = True,
)

# predict=True strictly isolates the end of the time series (the remaining 20% future)
validation = TimeSeriesDataSet.from_dataset(
    training, df_val,
    predict            = True, # FIXED: Must be True to evaluate only unseen future
    stop_randomization = True,
    allow_missing_timesteps = True,
)

train_loader = training.to_dataloader(
    train=True, batch_size=BATCH_SIZE, num_workers=4, shuffle=True
)
val_loader = validation.to_dataloader(
    train=False, batch_size=BATCH_SIZE, num_workers=4
)

# --- ADJUSTED TFT ARCHITECTURE ---
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate              = LEARNING_RATE,
    hidden_size                = 16,         
    attention_head_size        = 2,          
    dropout                    = 0.3,        # Increased to limit overfitting
    hidden_continuous_size     = 8,         
    loss                       = QuantileLoss(),
    optimizer                  = 'adam',
    reduce_on_plateau_patience = 3,
    log_interval               = 10,
)

print(f"Batches train : {len(train_loader)}")
print(f"Batches val   : {len(val_loader)}")
print(f"Parameters    : {tft.size()/1e3:.1f}k")

# --- ADJUSTED CALLBACKS ---
os.makedirs('model', exist_ok=True)

early_stop = EarlyStopping(
    monitor  = 'val_loss',
    patience = 10,  # Increased to 10 to give the model time to converge/rebound
    mode     = 'min',
    verbose  = True,
)

checkpoint_best = ModelCheckpoint(
    dirpath    = 'model/',
    filename   = 'tft_best',
    monitor    = 'val_loss',
    mode       = 'min',
    save_top_k = 1,
)

checkpoint_last = ModelCheckpoint(
    dirpath   = 'model/',
    filename  = 'tft_last',
    save_last = True,
)

# --- TRAINER ---
trainer = Trainer(
    max_epochs          = MAX_EPOCHS,
    accelerator         = 'gpu' if torch.cuda.is_available() else 'cpu',
    devices             = 1,
    gradient_clip_val   = 0.1,
    callbacks           = [early_stop, checkpoint_best, checkpoint_last],
    enable_progress_bar = True,
    log_every_n_steps   = 10,
)

# --- LAUNCH ---
last_ckpt = 'model/tft_last.ckpt'

if args.resume and os.path.exists(last_ckpt):
    print(f"\nResuming from : {last_ckpt}")
    original_load = torch.load
    torch.load = lambda *a, **kw: original_load(*a, **{**kw, 'weights_only': False})
    try:
        trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader, ckpt_path=last_ckpt)
    finally:
        torch.load = original_load
else:
    if args.resume:
        print("No checkpoint found — training from scratch.")
    print("\nTraining TFT...")
    trainer.fit(tft, train_dataloaders=train_loader, val_dataloaders=val_loader)

print("=" * 55)
print(f"Best checkpoint  : model/tft_best.ckpt")
print(f"Last checkpoint  : model/tft_last.ckpt")
print(f"Best val_loss    : {checkpoint_best.best_model_score}")
print("=" * 55)
print("\nTo resume later : python train_tft.py --resume")