import pandas as pd

# Chargement du dataset
df = pd.read_csv('data/vietnam_monthly_dataset.csv')

# 1. Vérifier quels sites ont survécu aux filtres
print("=== SITES DISPONIBLES ===")
print(df['site_name'].unique())
print("-" * 40)

# 2. Choisir le site que tu veux analyser
# (Remplace par le nom exact d'un des sites affichés ci-dessus)
site_choisi = 'Hoi_An_IA' 

# 3. Filtrer le dataset pour ne garder QUE ce site
df_site = df[df['site_name'] == site_choisi]

# 4. Afficher le top 10 des transects pour ce site spécifique
print(f"=== TOP 10 TRANSECTS POUR : {site_choisi} ===")
print(df_site['segment_id'].value_counts().head(10))