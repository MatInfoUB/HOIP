import argparse
from hoip.load_data import load_data
import hoip
from sklearn.metrics import r2_score
import datetime
import os

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
    model = hoip.RegressionModel(options)
    model.compile()
    model.nn.fit(x=[X_train, one_hots_train], y=y_train, epochs=options.n_iter,
          validation_data=([X_test, one_hots_test], y_test))

    y_predict = model.nn.predict([X_train, one_hots_train])
    print('Training R2 is: ', r2_score(y_train, y_predict))

    y_predict = model.nn.predict([X_test, one_hots_test])
    print('Testing R2 is: ', r2_score(y_test, y_predict))

    today = datetime.datetime.today().strftime('%m_%d_%Y')
    result_dir = 'Results'
    filename = 'regression_' + today + '_' + options.output + '.h5'
    filename = os.path.join(result_dir, filename)
    model.nn.save(filename)

if __name__ == '__main__':
    main()