import pandas as pd
import os
import json
import warnings
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import shap
warnings.filterwarnings('ignore')

# --- 1. CONFIGURATION ---
DATA_PATH = 'data/vietnam_ml_dataset.csv'
MODEL_DIR = 'models'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set a number to sample (speeds up SHAP significantly), or None to use all data
SAMPLE_SIZE = 2000

print("1. Loading and preparing data...")
df = pd.read_csv(DATA_PATH)
df['date'] = pd.to_datetime(df['date'])

# --- 2. RECREATE CYCLICAL TIME TRANSFORMATIONS ---
# We must recreate the exact same features used during training
df['month_sin'] = np.sin(2 * np.pi * df['date'].dt.month / 12.0)
df['month_cos'] = np.cos(2 * np.pi * df['date'].dt.month / 12.0)

df = df.sort_values(by=['segment_id', 'date'])

target = 'cross_distance_m'

# Drop missing targets
df_clean = df.dropna(subset=[target]).copy()

# Split into coast and river zones
df_coast = df_clean[df_clean['in_river_zone'] == 0].copy()
df_river = df_clean[df_clean['in_river_zone'] == 1].copy()

# --- 3. LOADING TRAINING FEATURES ---
print("2. Loading training feature order...")
with open(f'{MODEL_DIR}/training_features.json', 'r') as f:
    training_features = json.load(f)

# Select only the features the model was trained on
X_coast = df_coast[training_features].copy()
X_river = df_river[training_features].copy()

# --- 4. LOADING MODELS ---
print("3. Loading trained models...")
# Note: enable_categorical is removed since we dropped all categorical features
model_coast = XGBRegressor(tree_method='hist')
model_coast.load_model(f'{MODEL_DIR}/coast_model.json')

model_river = XGBRegressor(tree_method='hist')
model_river.load_model(f'{MODEL_DIR}/river_model.json')

# --- 5. SHAP ANALYSIS ---
def run_shap_analysis(model, X, name, sample_size=None):
    print(f"\n4. Computing SHAP values for '{name}' model...")

    if sample_size is not None and len(X) > sample_size:
        X_used = X.sample(sample_size, random_state=42)
        print(f"   -> Using a sample of {sample_size} / {len(X)} rows for speed.")
    else:
        X_used = X
        print(f"   -> Using full dataset ({len(X)} rows).")

    # TreeExplainer is highly optimized for XGBoost
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X_used)

    # Summary plot (beeswarm)
    plt.figure()
    shap.summary_plot(shap_values, X_used, show=False)
    plt.title(f"SHAP Summary - {name} model")
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_summary_{name}.png', dpi=150)
    plt.close()
    print(f"   -> Saved {PLOTS_DIR}/shap_summary_{name}.png")

    # Bar plot (feature importance)
    plt.figure()
    shap.summary_plot(shap_values, X_used, plot_type="bar", show=False)
    plt.title(f"SHAP Feature Importance - {name} model")
    plt.tight_layout()
    plt.savefig(f'{PLOTS_DIR}/shap_importance_{name}.png', dpi=150)
    plt.close()
    print(f"   -> Saved {PLOTS_DIR}/shap_importance_{name}.png")

    return shap_values, X_used

# Run the analysis for both environments
shap_values_coast, X_coast_used = run_shap_analysis(model_coast, X_coast, "coast", SAMPLE_SIZE)
shap_values_river, X_river_used = run_shap_analysis(model_river, X_river, "river", SAMPLE_SIZE)

print(f"\nDone. All SHAP plots saved in '{PLOTS_DIR}/' directory.")