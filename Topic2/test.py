import pandas as pd
df = pd.read_csv('data/vietnam_monthly_dataset.csv')
print(df['segment_id'].value_counts().head(30))