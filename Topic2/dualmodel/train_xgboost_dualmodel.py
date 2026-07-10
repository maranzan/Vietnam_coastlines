import pandas as pd
import os
import json
import warnings
import numpy as np
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
DATA_PATH = 'data/vietnam_ml_dataset.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

# We will use a time-based split instead of a random group split
TEST_SPLIT_RATIO = 0.2 
RANDOM_STATE = 42

print("1. Loading and preparing data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# Sort strictly by time to prevent data leakage during temporal split
df = df.sort_values(by=['date', 'segment_id'])

# --- 2. CYCLICAL TIME TRANSFORMATIONS ---
# Transform 'month' into continuous cyclical features (Sine / Cosine)
# This replaces the redundant 'month', 'year', and 'season' variables
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12.0)

# TARGET: absolute shoreline position
target = 'cross_distance_m'

# Drop rows with missing target
df_clean = df.dropna(subset=[target]).copy()

# --- 3. FEATURE SELECTION ---
# Based on SHAP analysis and Professor's guidelines to prevent spatial overfitting
# Keeping physical parameters (indices: 1, 4, 5, 6, 8, 11, 12, 13, 15, 16, 17, 19)
selected_physical_features = [
    'dist_river_m', 'orientation_deg', 'elevation_m', 'slope_deg',
    'Hs_mean_7d', 'Hs_max_7d', 'wind_mean_7d', 'wave_dir_deg',
    'wave_period_s', 'wind_dir_deg', 'days_since_prev', 'tide_height_m'
]

# Combine physical features with our newly created time variables
training_features = selected_physical_features + ['month_sin', 'month_cos']

# --- 4. TIME-SERIES SPLIT ---
print("\n2. Splitting data temporally (Train on past, Test on future)...")

def time_series_split(df_zone, test_ratio):
    """Splits the dataset sequentially to respect the arrow of time."""
    n_samples = len(df_zone)
    split_idx = int(n_samples * (1 - test_ratio))
    
    # Because the dataframe is already sorted by date, we just slice it
    df_train = df_zone.iloc[:split_idx].copy()
    df_test = df_zone.iloc[split_idx:].copy()
    return df_train, df_test

df_coast_all = df_clean[df_clean['in_river_zone'] == 0].copy()
df_river_all = df_clean[df_clean['in_river_zone'] == 1].copy()

df_coast_train, df_coast_test = time_series_split(df_coast_all, TEST_SPLIT_RATIO)
df_river_train, df_river_test = time_series_split(df_river_all, TEST_SPLIT_RATIO)

print(f"   Coast : {len(df_coast_train)} rows train / {len(df_coast_test)} rows test")
print(f"   River : {len(df_river_train)} rows train / {len(df_river_test)} rows test")

# Save exact column order for future predictions
with open(f'{MODEL_DIR}/training_features.json', 'w') as f:
    json.dump(training_features, f)

# Save test sets
df_coast_test.to_csv(f'{MODEL_DIR}/test_coast.csv', index=False)
df_river_test.to_csv(f'{MODEL_DIR}/test_river.csv', index=False)

# --- 5. MODEL TRAINING ---
print("\n3. Training Coastal and River models...")

params_coast = {
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 6,
    'subsample': 0.8, 'colsample_bytree': 0.8, 
    'tree_method': 'hist', 'random_state': RANDOM_STATE
    # 'enable_categorical': True is removed since we have only numerical/continuous data now
}
model_coast = XGBRegressor(**params_coast)
model_coast.fit(df_coast_train[training_features], df_coast_train[target])

params_river = {
    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 5,
    'subsample': 0.8, 'colsample_bytree': 0.8, 
    'tree_method': 'hist', 'random_state': RANDOM_STATE
}
model_river = XGBRegressor(**params_river)
model_river.fit(df_river_train[training_features], df_river_train[target])

# --- 6. EVALUATION ---
print("\n4. Evaluating on future temporal data...")

def evaluate(model, df_test, name):
    X_test = df_test[training_features]
    y_test = df_test[target]
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"   {name}: MAE = {mae:.2f} m | R2 = {r2:.3f}")

evaluate(model_coast, df_coast_test, "Coast (Future Time Block)")
evaluate(model_river, df_river_test, "River (Future Time Block)")

# --- 7. SAVING MODELS ---
print("\n5. Saving models to disk...")
model_coast.save_model(f'{MODEL_DIR}/coast_model.json')
model_river.save_model(f'{MODEL_DIR}/river_model.json')
print("Models saved successfully.")