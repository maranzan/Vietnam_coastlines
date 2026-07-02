import pandas as pd
import os
import json
import warnings
import matplotlib.pyplot as plt
from xgboost import XGBRegressor
import shap

warnings.filterwarnings('ignore')

# --- CONFIGURATION ---
DATA_PATH = 'data/vietnam_ml_dataset.csv'
MODEL_DIR = 'models'
os.makedirs(MODEL_DIR, exist_ok=True)

print("1. Loading and preparing data for training...")
df = pd.read_csv(DATA_PATH)

df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df = df.sort_values(by=['segment_id', 'date'])

df['lag_1_change'] = df.groupby('segment_id')['shoreline_change_m'].shift(1)
df['rolling_3_change'] = df.groupby('segment_id')['shoreline_change_m'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

target = 'shoreline_change_m'

df['site_name'] = df['site_name'].astype('category')
df['satellite'] = df['satellite'].astype('category')
cols_to_drop = ['segment_id', 'date', 'cross_distance_m', 'in_river_zone', target]

df_train = df.dropna(subset=[target, 'lag_1_change', 'rolling_3_change'])
df_coast = df_train[df_train['in_river_zone'] == 0].copy()
df_river = df_train[df_train['in_river_zone'] == 1].copy()

# Save the exact feature order for inference to prevent mismatch errors
training_features = df_coast.drop(columns=cols_to_drop).columns.tolist()
with open(f'{MODEL_DIR}/training_features.json', 'w') as f:
    json.dump(training_features, f)

# --- MODEL TRAINING ---
print("\n2. Training Coastal and River models...")
params_coast = {
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 
    'subsample': 0.8, 'colsample_bytree': 0.8, 'enable_categorical': True, 
    'tree_method': 'hist', 'random_state': 42
}
model_coast = XGBRegressor(**params_coast)
model_coast.fit(df_coast.drop(columns=cols_to_drop), df_coast[target])

params_river = {
    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 4, 
    'subsample': 0.8, 'colsample_bytree': 0.8, 'enable_categorical': True, 
    'tree_method': 'hist', 'random_state': 42
}
model_river = XGBRegressor(**params_river)
model_river.fit(df_river.drop(columns=cols_to_drop), df_river[target])

# --- SAVING MODELS ---
print("\n3. Saving models to disk...")
model_coast.save_model(f'{MODEL_DIR}/coast_model.json')
model_river.save_model(f'{MODEL_DIR}/river_model.json')
print(f"Models saved in '{MODEL_DIR}/' directory.")

# --- SHAP EXPLAINABILITY ---
print("\n4. Generating SHAP Explainability Plot (Coastal Model)...")
explainer_coast = shap.TreeExplainer(model_coast)
X_sample = df_coast.drop(columns=cols_to_drop).sample(min(1000, len(df_coast)), random_state=42)
shap_values_coast = explainer_coast(X_sample)

plt.figure(figsize=(10, 6))
shap.summary_plot(shap_values_coast, X_sample, show=False)
plt.title("SHAP Feature Importance (Coastal Model)")
plt.tight_layout()
plt.show()

print("\nTraining Phase Completed!")