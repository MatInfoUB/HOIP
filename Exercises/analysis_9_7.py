import matplotlib.pyplot as plt
import pandas as pd
from pca import pca
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import numpy as np
import os
from datetime import datetime
# import umap

figdir = '../figs'
figdir = os.path.join(figdir, datetime.today().strftime('%Y_%m_%d'))

if not os.path.isdir(figdir):
       os.mkdir(figdir)

data = pd.read_csv('../Data/features_dataframe_for_Arpan_for_HOIPs.csv')

cols = ['organic_globularity',
       'organic_surface_area', 'organic_volume',
       'length_est_by_volume',
       'X_atoms_B_neighbor_percent_stdev', 'X_atoms_B_neighbor_percent_mean',
       'X_atoms_H_neighbor_percent_stdev', 'X_atoms_H_neighbor_percent_mean',
       'X_atoms_any_organic_neighbor_percent_stdev',
       'X_atoms_any_organic_neighbor_percent_mean',
       'B_atom_H_neighbor_percent', 'B_atom_X_neighbor_percent',
       'B_atom_any_organic_neighbor_percent',
       'organic_molecule_B_neighbor_percent',
       'organic_molecule_X_neighbor_percent',
       'organic_molecule_any_organic_neighbor_percent']

data = data.drop(columns=data.columns[0])
dtypes = data.dtypes

num_cols = dtypes[dtypes == 'float64'].index
num_data = data[cols]
# num_data = data[num_cols]

std = StandardScaler()
num_data = pd.DataFrame(std.fit_transform(num_data),
                        columns=num_data.columns)

atoms = ['B_atom', 'X_atom', 'organic molecule']

for atom in atoms:
       num_data = num_data.set_index(data[atom])
       # sns.heatmap(num_data.corr())

       model = pca(n_components=0.99)
       results = model.fit_transform(num_data)

       fig, ax = model.biplot(n_feat=5, SPE=True)
       outliers = results['PC'][results['outliers'].y_bool_spe]
       for i, row in outliers.iterrows():
              ax.text(row['PC1'], row['PC2'], i, fontsize=6)

       figfile = os.path.join(figdir, 'biplot_'+ atom + '.png')
       plt.savefig(figfile, bbox_inches='tight')
       plt.close()

loadings = results['loadings']
loadings[np.abs(loadings) < 1e-3] = 0

sns.clustermap(num_data.corr())

# model = umap.UMAP()
# embedding = pd.DataFrame(model.fit_transform(num_data), columns=['x', 'y'])
# embedding['color'] = data[atoms[1]].values