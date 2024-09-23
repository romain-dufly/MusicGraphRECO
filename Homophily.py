
## Homophily in artist graph ##

'''We aim to visualize the closeness of collaborating artists'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# Read in data
collabs = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/edges_collabs.csv', sep=';')
artists = pd.read_csv('/Users/romaindufly/Desktop/Projet MODAL/embeddings.csv', sep=';')

artists['embedding'] = artists['embedding'].apply(lambda x: np.fromstring(x[1:-1], sep=' '))
artists_dict = {}
for i in range(len(artists)):
    artists_dict[artists['id'][i]] = artists['embedding'][i]

# Generate distances between collaborating artists
distances = []
for i in range(len(collabs)):
    artist1 = collabs['Source'][i]
    artist2 = collabs['Target'][i]
    distance = 1 - cosine_similarity([artists_dict[artist1]], [artists_dict[artist2]])[0][0]
    if distance != 0 :
        distances.append(distance)

# Generate distances between random pairs of artists
random_distances = []
for i in range(len(distances)):
    artist1 = np.random.choice(list(artists_dict.keys()))
    artist2 = np.random.choice(list(artists_dict.keys()))
    distance = 1 - cosine_similarity([artists_dict[artist1]], [artists_dict[artist2]])[0][0]
    if distance != 0 :
        random_distances.append(distance)

# Plot the distributions of distances
h, edges = np.histogram(distances, bins=100)
plt.stairs(h, edges, label='Collaborations')
h, edges = np.histogram(random_distances, bins=100)
plt.stairs(h, edges, label='Random pairs')
# Add vertical lines for the median distances
plt.axvline(np.median(distances), color='blue', linestyle='dashed', linewidth=1)
plt.axvline(np.median(random_distances), color='orange', linestyle='dashed', linewidth=1)
plt.xlabel('Distance')
plt.ylabel('Number of collaborations')
plt.title('Distribution of distances between collaborating artists')
plt.legend()
plt.show()