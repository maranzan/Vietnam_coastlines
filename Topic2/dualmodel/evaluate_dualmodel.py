import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = 'data/vietnam_ml_dataset.csv'
MODEL_DIR = 'models'

print("1. Loading Data and Pre-Trained Models...")
# Load feature names used for training
with open(f'{MODEL_DIR}/training_features.json', 'r') as f:
    training_features = json.load(f)

# Load XGBoost models
model_coast = XGBRegressor()
model_coast.load_model(f'{MODEL_DIR}/coast_model.json')

model_river = XGBRegressor()
model_river.load_model(f'{MODEL_DIR}/river_model.json')

# Prepare data (exactly as done during training)
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by=['segment_id', 'date'])

df['lag_1_change'] = df.groupby('segment_id')['shoreline_change_m'].shift(1)
df['rolling_3_change'] = df.groupby('segment_id')['shoreline_change_m'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Convert text categories for XGBoost compatibility
df['site_name'] = df['site_name'].astype('category')
df['satellite'] = df['satellite'].astype('category')

# Drop rows where the target or features are missing (needed for comparison)
target = 'shoreline_change_m'
df_eval = df.dropna(subset=[target, 'lag_1_change', 'rolling_3_change']).copy()

print("2. Running Predictions on Historical Data...")
mask_coast = df_eval['in_river_zone'] == 0
mask_river = df_eval['in_river_zone'] == 1

# The model predicts "blindly" based on each month's conditions
if mask_coast.sum() > 0:
    df_eval.loc[mask_coast, 'pred_change'] = model_coast.predict(df_eval.loc[mask_coast, training_features])
if mask_river.sum() > 0:
    df_eval.loc[mask_river, 'pred_change'] = model_river.predict(df_eval.loc[mask_river, training_features])

print("\n3. Calculating Performance Metrics...")

def print_metrics(name, y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"=== {name} ===")
    print(f"  RMSE : {rmse:.2f} m (Root Mean Squared Error)")
    print(f"  MAE  : {mae:.2f} m (Mean Absolute Error)")
    print(f"  R²   : {r2:.3f}  (Prediction Quality, max=1.0)\n")

# Display evaluation scores
print_metrics("GLOBAL PERFORMANCE (Coast + River)", df_eval[target], df_eval['pred_change'])
if mask_coast.sum() > 0:
    print_metrics("COASTAL MODEL ONLY", df_eval.loc[mask_coast, target], df_eval.loc[mask_coast, 'pred_change'])
if mask_river.sum() > 0:
    print_metrics("RIVER/ESTUARY MODEL ONLY", df_eval.loc[mask_river, target], df_eval.loc[mask_river, 'pred_change'])

print("4. Generating Evaluation Plot...")
plt.figure(figsize=(10, 8))

# Scatter plot (Actual vs. Predicted)
plt.scatter(df_eval.loc[mask_coast, target], df_eval.loc[mask_coast, 'pred_change'], 
            alpha=0.4, label='Coastal', color='blue', s=15)
plt.scatter(df_eval.loc[mask_river, target], df_eval.loc[mask_river, 'pred_change'], 
            alpha=0.4, label='River/Estuary', color='green', s=15)

# Dashed red line representing the "perfect" prediction (y = x)
min_val = min(df_eval[target].min(), df_eval['pred_change'].min())
max_val = max(df_eval[target].max(), df_eval['pred_change'].max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction (Actual = Predicted)', linewidth=2)

plt.title("XGBoost Performance: Actual vs. Predicted Values", fontsize=14)
plt.xlabel("Actual Shoreline Movement (m/month)", fontsize=12)
plt.ylabel("AI Predicted Movement (m/month)", fontsize=12)
plt.axhline(0, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.legend()
plt.grid(True, linestyle=':', alpha=0.7)
plt.tight_layout()
plt.show()