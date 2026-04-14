import pickle
import matplotlib.pyplot as plt
import os

filepath = os.path.join(os.getcwd(), 'data', 'Hoi_An_IA', 'Hoi_An_IA_output.pkl')

with open(filepath, 'rb') as f:
    output = pickle.load(f)


plt.figure(figsize=(10, 5))
for i in range(len(output['shorelines'])):
    sl = output['shorelines'][i]
    plt.plot(sl[:,0], sl[:,1], '.', markersize=1)

plt.title("Superposition de toutes les lignes de côte extraites (Hội An)")
plt.xlabel("Easting (m)")
plt.ylabel("Northing (m)")
plt.axis('equal')
plt.show()