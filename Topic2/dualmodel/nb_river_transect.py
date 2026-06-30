import pandas as pd

# Remplace par le bon chemin si nécessaire (ex: data/vietnam_ml_dataset.csv)
fichier_csv = 'data/vietnam_ml_dataset.csv'

print(f"Chargement du fichier : {fichier_csv}...\n")
df = pd.read_csv(fichier_csv)

# Isoler un seul enregistrement par transect
transects_uniques = df.drop_duplicates(subset=['segment_id'])

# Calculs
total_transects = len(transects_uniques)
nb_riviere = int(transects_uniques['in_river_zone'].sum())
nb_cote = total_transects - nb_riviere

# Affichage des résultats
print("-" * 40)
print("📊 RÉPARTITION DES TRANSECTS")
print("-" * 40)
print(f"Total de transects uniques : {total_transects}")
print(f"🌊 Côtiers (in_river_zone = 0) : {nb_cote} ({(nb_cote/total_transects)*100:.1f}%)")
print(f"🌿 Rivières (in_river_zone = 1) : {nb_riviere} ({(nb_riviere/total_transects)*100:.1f}%)")
print("-" * 40)