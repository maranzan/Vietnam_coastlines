import pandas as pd
import numpy as np
import os
import warnings
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import folium
from folium.plugins import MarkerCluster

warnings.filterwarnings('ignore')

print("1. Loading dataset and engineering features...")
df = pd.read_csv('data/vietnam_monthly_dataset.csv')

# --- FEATURE ENGINEERING (PERFORMANCE BOOST) ---
# Convert YearMonth to datetime to extract seasonal patterns
df['date'] = pd.to_datetime(df['YearMonth'])
df['month'] = df['date'].dt.month

# Sort data by location and time to safely calculate historical trends
df = df.sort_values(by=['segment_id', 'date'])

# Create historical features (Lag and Rolling Mean)
df['lag_1_change'] = df.groupby('segment_id')['monthly_change_m'].shift(1)
df['rolling_3_change'] = df.groupby('segment_id')['monthly_change_m'].transform(
    lambda x: x.shift(1).rolling(window=3, min_periods=1).mean()
)

# Set correct data types
target = 'monthly_change_m'
df['site_name'] = df['site_name'].astype('category')

# Define columns to exclude from training features
cols_to_drop = [
    'segment_id', 'YearMonth', 'date', 
    'cross_distance_m', 'in_river_zone', target
]

# --- DATASET SEPARATION ---
df_coast = df[df['in_river_zone'] == 0].copy()
df_river = df[df['in_river_zone'] == 1].copy()

print(f"Training distribution:\nCoastal Model: {len(df_coast)} obs\nRiver Model: {len(df_river)} obs")

# --- EVALUATION FUNCTION ---
def evaluate_model(data, model_name, xgb_params):
    print(f"\n{'-'*40}\nTRAINING: {model_name.upper()}\n{'-'*40}")
    
    X = data.drop(columns=cols_to_drop)
    y = data[target]
    
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    model = XGBRegressor(**xgb_params)
    
    scores_r2, scores_rmse = [], []
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
        X_test, y_test   = X.iloc[test_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        
        scores_rmse.append(rmse)
        scores_r2.append(r2)
        print(f"Fold {fold+1} | RMSE: {rmse:.2f} | R2: {r2:.2f}")

    print(f"\n> {model_name} MEAN RMSE: {np.mean(scores_rmse):.2f}")
    print(f"> {model_name} MEAN R2:   {np.mean(scores_r2):.2f}")
    
    final_model = XGBRegressor(**xgb_params)
    final_model.fit(X, y)
    return final_model, X

# --- TRAINING ---
# Added colsample_bytree and subsample to prevent overfitting on the new features
params_coast = {
    'n_estimators': 500, 'learning_rate': 0.05, 'max_depth': 5, 
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'enable_categorical': True, 'tree_method': 'hist', 'random_state': 42
}

params_river = {
    'n_estimators': 400, 'learning_rate': 0.05, 'max_depth': 4, 
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'enable_categorical': True, 'tree_method': 'hist', 'random_state': 42
}

model_coast, X_coast = evaluate_model(df_coast, "Coastal Model", params_coast)
model_river, X_river = evaluate_model(df_river, "River Model", params_river)

print("\n2. Generating optimized Folium map...")

# --- MAP GENERATION ---
df_coast['prediction'] = model_coast.predict(X_coast)
df_river['prediction'] = model_river.predict(X_river)
df_map = pd.concat([df_coast, df_river])

max_points = 15000
if len(df_map) > max_points:
    df_map = df_map.sample(n=max_points, random_state=42)

m = folium.Map(location=[16.0, 106.0], zoom_start=6, tiles='CartoDB positron')
marker_cluster = MarkerCluster(name="Erosion Predictions").add_to(m)

def get_color(val):
    if val < -1.0: return 'darkred'
    if val < 0.0: return 'orange'
    if val == 0.0: return 'gray'
    if val <= 1.0: return 'lightgreen'
    return 'darkgreen'

for idx, row in df_map.iterrows():
    zone_type = "Estuary/River" if row['in_river_zone'] == 1 else "Coast"
    
    popup_html = f"""
    <div style="font-family: Arial; min-width: 200px;">
        <h4 style="margin-bottom: 5px;">{row['site_name']}</h4>
        <b>Zone:</b> {zone_type}<br>
        <b>Actual Change:</b> {row['monthly_change_m']:.2f} m<br>
        <b>ML Prediction:</b> {row['prediction']:.2f} m
    </div>
    """
    
    folium.CircleMarker(
        location=[row['latitude'], row['longitude']],
        radius=6,
        color=get_color(row['prediction']),
        fill=True,
        fill_color=get_color(row['prediction']),
        fill_opacity=0.8,
        popup=folium.Popup(popup_html, max_width=300),
        tooltip=f"{row['site_name']} details"
    ).add_to(marker_cluster)

output_file = 'data/maps/predictions_erosion_vietnam_cluster.html'
os.makedirs(os.path.dirname(output_file), exist_ok=True)
m.save(output_file)

print(f"Map successfully saved to '{output_file}'.")