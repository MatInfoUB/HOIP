import argparse
import numpy as np
from hoip.load_data import load_data
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--output", default='relative energy1', help='Output Parameter')
parser.add_argument("--n_att_layer", default=1, help='Number of Attention Layer')
parser.add_argument("--n_fc", default=1, help='Number of fc Layers')
parser.add_argument("--n_iter", default=100, help='Number of Iterations')
parser.add_argument("--optimizer", default='Adam', help='Optimizer')
parser.add_argument("--loss", default='mse', help='Loss function')

options = parser.parse_args()


def main():

    X_train, X_test, one_hots_train, \
    one_hots_test, y_train, y_test, inds = load_data(options, return_inds=True)
    model_name = [m for m in os.listdir('Results') if 'h5' in m][0]
    model = load_model(os.path.join('Results', model_name))

    cluster = pd.read_excel('Results/cluster_member.xlsx')

    y_predict = model.predict([X_train, one_hots_train])
    print('Training R2 is: ', r2_score(y_train, y_predict))
    residual = y_train - y_predict.reshape(len(y_train))
    fig, ax = plt.subplots()
    ax.scatter(y_train, residual, c='cyan')

    residual = pd.DataFrame({'Observed': y_train, 'Predicted':
        y_predict.reshape(len(y_train))}, index=y_train.index)
    hs_nn_train = []
    for col in cluster.columns:
        s = cluster[col].isna()
        s = cluster[col][~s].astype(int)
        res_ind = [ind for ind in y_train.index if ind in s]
        res_ind = residual.loc[res_ind]
        hs_nn_train.append(r2_score(res_ind['Observed'], res_ind['Predicted']))

    y_predict = model.predict([X_test, one_hots_test])
    print('Testing R2 is: ', r2_score(y_test, y_predict))
    residual = y_test - y_predict.reshape(len(y_test))
    ax.scatter(y_test, residual, c='b')
    ax.set_ylim([-0.25, 0.45])
    plt.show()

    residual = pd.DataFrame({'Observed': y_test, 'Predicted':
        y_predict.reshape(len(y_test))}, index=y_test.index)
    hs_nn_test = []
    for col in cluster.columns:
        s = cluster[col].isna()
        s = cluster[col][~s].astype(int)
        res_ind = [ind for ind in y_test.index if ind in s]
        res_ind = residual.loc[res_ind]
        hs_nn_test.append(r2_score(res_ind['Observed'], res_ind['Predicted']))

    print(hs_nn_train)
    print(hs_nn_test)
    labels = dict()
    for col in cluster.columns:
        s = cluster[col].isna()
        s = cluster[col][~s].astype(int)
        for i in s:
            labels[i] = col
    labels_new = [labels[k] for k in inds]


if __name__ == '__main__':
    main()