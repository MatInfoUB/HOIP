import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import hdbscan
from sklearn.preprocessing import StandardScaler
import seaborn as sns
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