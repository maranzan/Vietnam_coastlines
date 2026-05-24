import pandas as pd
import numpy as np
import os

# 1. Load the long-format data
df = pd.read_csv('data/dataset_ready_for_ia.csv')

def create_sequences(data, seq_length):
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

seq_length = 10 
X_total, y_total = [], []

# 2. Create sequences for each unique transect
for tid in df['unique_id'].unique():
    transect_data = df[df['unique_id'] == tid]['distance_scaled'].values
    
    if len(transect_data) > seq_length:
        X_ts, y_ts = create_sequences(transect_data, seq_length)
        X_total.append(X_ts)
        y_total.append(y_ts)

# 3. FIX: Use concatenate instead of vstack
X = np.concatenate(X_total, axis=0)
y = np.concatenate(y_total, axis=0)

# 4. Final Reshape for LSTM: [Samples, Time_Steps, Features]
X = X.reshape((X.shape[0], X.shape[1], 1))
y = y.reshape((y.shape[0], 1))

# 5. Save
os.makedirs('data', exist_ok=True)
np.save('data/X_train.npy', X)
np.save('data/y_train.npy', y)

print(f"Success! Final training set shape: {X.shape}")
print(f"Total samples available for training: {X.shape[0]}")