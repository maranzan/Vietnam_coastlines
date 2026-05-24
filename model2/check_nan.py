import numpy as np
import pandas as pd

print("--- DIAGNOSTIC SÉCURITÉ ---")

# 1. Vérification du CSV final
df = pd.read_csv('data/dataset_ready_for_ia.csv')
print("NaNs dans le CSV :", df[['distance', 'wave_height', 'wind_speed', 'wave_period']].isna().sum().sum())

# 2. Vérification des fichiers de séquences numpy (.npy)
try:
    X = np.load('data/X_train_multi.npy')
    y = np.load('data/y_train_multi.npy')
    
    print(f"Shape de X: {X.shape}")
    print("Y a-t-il des NaNs dans X_train_multi ?", np.isnan(X).any())
    print("Y a-t-il des NaNs dans y_train_multi ?", np.isnan(y).any())
    print("Y a-t-il des valeurs infinies (inf) dans X ?", np.isinf(X).any())
    
    # Affichage des premières valeurs pour voir à quoi ça ressemble
    print("\nExemple de première séquence (X[0]) :\n", X[0])
except FileNotFoundError:
    print("Fichiers .npy introuvables.")