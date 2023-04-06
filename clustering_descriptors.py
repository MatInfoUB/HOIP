import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import umap

# Data set
df1 = pd.read_csv('Data/2021_09_21_HOIP_dataframe_for_Arpan_and_Ruhil.csv',
                  usecols=lambda x: 'Unnamed' not in x,
                  header=0
                 )

# text file
with open('Data/descriptors_2.txt') as f:
    lines = f.read().splitlines() # Read lines and remove '\n'
lines = lines[1:-7]

# Only 17 descriptors
df2 = df1[lines]

## UMAP calculation
# Scale data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df2)

# UMAP parameters
min_dist = 0.2 # min_dist parameter controls how tightly UMAP is allowed to pack points together, local = 0.2, global = 1
n_neighbors = 10 # balances local versus global structure in the data, local = 10, global = 100
n_components = 3
metric = 'euclidean'
random_state = 2424

mapper = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors,
                   n_components=n_components, random_state=random_state, output_dens=True)
embedding = mapper.fit_transform(scaled_data)

# Clustering on UMAP embedding
clusterer = hdbscan.HDBSCAN(min_samples=10,
                            min_cluster_size=50,
                           )

labels = clusterer.fit_predict(embedding[0])

