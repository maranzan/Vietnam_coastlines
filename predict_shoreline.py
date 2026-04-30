import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

# 1. Load the model, the scaler, and the data
from model_erosion import ErosionLSTM # Assuming class is in your script

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ErosionLSTM(input_size=1, hidden_size=64, num_layers=2, output_size=1)
model.load_state_dict(torch.load('model/erosion_lstm_v1.pth'))
model.to(device)
model.eval()

with open('data/scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

df = pd.read_csv('data/dataset_ready_for_ia.csv')

# 2. Pick a random transect to test (ex: one from Hoi An)
target_id = df['unique_id'].unique()[0] 
test_data = df[df['unique_id'] == target_id]['distance_scaled'].values

# Take the last 10 points to predict the future
input_seq = torch.from_numpy(test_data[-10:]).float().view(1, 10, 1).to(device)

with torch.no_grad():
    prediction_scaled = model(input_seq)
    # Convert back to meters using the inverse scaler
    prediction_meters = scaler.inverse_transform(prediction_scaled.cpu().numpy())
    actual_meters = scaler.inverse_transform(test_data[-1:].reshape(-1, 1))

print(f"📊 Result for {target_id}:")
print(f"Last known position: {actual_meters[0][0]:.2f} m")
print(f"AI Predicted next position: {prediction_meters[0][0]:.2f} m")

# 3. Quick Plot
plt.plot(scaler.inverse_transform(test_data[-20:].reshape(-1,1)), label="Past (Last 20 points)")
plt.scatter(20, prediction_meters, color='red', label="AI Prediction")
plt.title(f"Shoreline Prediction for {target_id}")
plt.ylabel("Distance (m)")
plt.legend()
plt.show()