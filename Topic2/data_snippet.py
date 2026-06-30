import pandas as pd

# Remplace par le chemin correct de ton fichier
chemin_fichier = 'data/vietnam_monthly_dataset.csv'

# Optionnel : Configurer Pandas pour afficher toutes les colonnes proprement
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("=" * 80)
print(f"  EXTRAIT DU DATASET ENRICHI : {chemin_fichier}")
print("=" * 80)

try:
    df = pd.read_csv(chemin_fichier)
    
    # Afficher les 5 premières lignes
    # On transposes (.T) souvent quand il y a beaucoup de colonnes 
    # pour que ça soit plus lisible dans un terminal
    print(df.head(5).to_markdown())
    
    print("\n" + "=" * 80)
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print("=" * 80)
    
except FileNotFoundError:
    print(f"Erreur : Le fichier {chemin_fichier} n'a pas été trouvé.")