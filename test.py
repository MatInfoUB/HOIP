import argparse
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
    one_hots_test, y_train, y_test = load_data(options)
    res_dir = os.path.join('Results', options.output)
    models = [m for m in os.listdir(res_dir) if 'h5' in m]
    model = load_model(os.path.join(res_dir, models[1]))

    y_predict = model.predict([X_train, one_hots_train])
    print('Training R2 is: ', r2_score(y_train, y_predict))
    residual = y_train - y_predict.reshape(len(y_train))
    fig, ax = plt.subplots()
    ax.scatter(y_train, residual, c='cyan')

    y_predict = model.predict([X_test, one_hots_test])
    print('Testing R2 is: ', r2_score(y_test, y_predict))
    residual = y_test - y_predict.reshape(len(y_test))
    ax.scatter(y_test, residual, c='b')
    ax.set_ylim([-0.25, 0.45])
    plt.show()

    cluster = pd.read_excel('Results/cluster_member.xlsx')


if __name__ == '__main__':
    main()