import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

df = pd.read_csv('data/vietnam_master_dataset.csv', parse_dates=['dates'])


ts_cols = [c for c in df.columns if c.startswith('TS_')]

print(f"Analysing {len(ts_cols)} transects on {df['site_name'].unique()} sites.")

missing_data = df[ts_cols].isnull().mean() * 100
plt.figure(figsize=(12, 4))
missing_data.plot(kind='bar', color='salmon')
plt.title("Percentage of missing data by Transect")
plt.ylabel("% of NaN")
plt.axhline(y=50, color='r', linestyle='--', label='Critical limit (50%)')
plt.legend()
plt.show()

site_to_plot = df['site_name'].unique()[0]
sample_df = df[df['site_name'] == site_to_plot].sort_values('dates')

plt.figure(figsize=(14, 6))
for ts in ts_cols[:5]:
    plt.plot(sample_df['dates'], sample_df[ts], label=ts, alpha=0.7)

plt.title(f"Evolution of the coast line - Site : {site_to_plot}")
plt.xlabel("Year")
plt.ylabel("Distance (m)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()


plt.figure(figsize=(10, 8))
corr = df[ts_cols[:20]].corr()
sns.heatmap(corr, cmap='coolwarm', center=0, annot=False)
plt.title("Correlation between Transects (Spatial proximity)")
plt.show()

recul_total = df[ts_cols].iloc[-1] - df[ts_cols].iloc[0]
print("\nTop 3 most erroded transects :")
print(recul_total.sort_values().head(3))