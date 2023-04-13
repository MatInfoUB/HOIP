import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
import seaborn as sns
from sklearn.model_selection import train_test_split
import umap


def load_data(options, return_data=False):

    data = pd.read_csv('Data/HOIP_dataframe.csv')
    descriptor_list = pd.read_csv('Data/descriptors_2.txt')
    descriptor_list = descriptor_list['Descriptors']
    data = data.set_index(data.file)

    # seperating the scalar
    X = data[descriptor_list[:-7]]
    X = StandardScaler().fit_transform(X)
    one_hots = data[descriptor_list[-7:]].astype('float')

    options.input_shape_1 = X.shape[1:]
    options.input_shape_2 = one_hots.shape[1:]

    y = data[options.output]

    # Train test split
    ind_train, ind_test = train_test_split(range(len(data)), random_state=42)
    X_train, X_test = X[ind_train], X[ind_test]
    ind_train, ind_test = train_test_split(data['file'].tolist(), random_state=42)
    y_train, y_test, one_hots_train, one_hots_test= y[ind_train], y[ind_test], one_hots.loc[ind_train], one_hots.loc[ind_test]

    if return_data:
        return X_train, X_test, one_hots_train, one_hots_test, y_train, y_test, data

    else:
        return X_train, X_test, one_hots_train, one_hots_test, y_train, y_test


def cluster_descriptors():
    # Data set
    df1 = pd.read_csv('Data/HOIP_dataframe.csv',
                      usecols=lambda x: 'Unnamed' not in x,
                      header=0
                      )

    # text file
    with open('Data/descriptors_2.txt') as f:
        lines = f.read().splitlines()  # Read lines and remove '\n'
    lines = lines[1:-7]

    # Only 17 descriptors
    df2 = df1[lines]

    ## UMAP calculation
    # Scale data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df2)

    # UMAP parameters
    min_dist = 0.2  # min_dist parameter controls how tightly UMAP is allowed to pack points together, local = 0.2, global = 1
    n_neighbors = 10  # balances local versus global structure in the data, local = 10, global = 100
    n_components = 3
    metric = 'euclidean'
    random_state = 2424

    mapper = umap.UMAP(min_dist=min_dist, n_neighbors=n_neighbors,
                       n_components=n_components, random_state=random_state, output_dens=True)
    mapper.fit(scaled_data)

    # Clustering on UMAP embedding
    clusterer = hdbscan.HDBSCAN(min_samples=10,
                                min_cluster_size=30,
                                )

    labels = clusterer.fit_predict(mapper.embedding_)
    fig, ax = plt.subplots(figsize=[8.65, 7.28])
    clusterer.condensed_tree_.plot(axis=ax, label_clusters=True, select_clusters=True)
    plt.show()

    return labels


def calculate_order(df, col=None):

    order_full = []
    file_list = []
    for unique_label in df.label.unique():
        temp = df[df['label'] == unique_label].sort_values(col, axis=0).reset_index()
        file_list += temp['file'].tolist()
        order_full += temp.index.tolist()

    order = pd.DataFrame({'file': file_list, 'order': order_full})
    order = order.sort_values(by='file')

    return order

