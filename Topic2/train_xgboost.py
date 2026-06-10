import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap

print("Chargement du dataset mensuel...")
df = pd.read_csv('data/vietnam_monthly_dataset.csv')

# Définition de la cible
target = 'monthly_change_m'
df['site_name'] = df['site_name'].astype('category')

# Retrait des variables inutiles pour l'IA (pour éviter la triche spatiale)
cols_to_drop = [
    'segment_id', 'YearMonth', 
    'latitude', 'longitude', 
    'cross_distance_m', 
    target
]

X = df.drop(columns=cols_to_drop)
y = df[target]

print("\nEntraînement et Validation Croisée...")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
model = XGBRegressor(
    n_estimators=500, 
    learning_rate=0.05, 
    max_depth=5, 
    enable_categorical=True, 
    tree_method='hist',
    random_state=42
)

scores_r2, scores_rmse = [], []

for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
    # Entraînement
    model.fit(X.iloc[train_idx], y.iloc[train_idx])
    
    # Prédiction
    preds = model.predict(X.iloc[test_idx])
    
    # Évaluation
    rmse = np.sqrt(mean_squared_error(y.iloc[test_idx], preds))
    r2 = r2_score(y.iloc[test_idx], preds)
    scores_rmse.append(rmse)
    scores_r2.append(r2)
    print(f"Fold {fold+1} | RMSE: {rmse:.2f} m/mois | R2: {r2:.2f}")

print("-" * 50)
print(f">>> MOYENNE RMSE : {np.mean(scores_rmse):.2f} m/mois")
print(f">>> MOYENNE R2   : {np.mean(scores_r2):.2f}")
print("-" * 50)

print("\nGénération de l'explication SHAP...")
# On entraîne une dernière fois sur toutes les données pour l'analyse finale
model.fit(X, y)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X)

shap.summary_plot(shap_values, X, plot_type="dot")