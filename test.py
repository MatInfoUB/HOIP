import argparse
import numpy as np
from hoip.load_data import *
import os
from tensorflow.keras.models import load_model
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set_context('talk')

parser = argparse.ArgumentParser()
parser.add_argument("--output", default='relative_energy1', help='Output Parameter')

options = parser.parse_args()


def main():

    options.output = ' '.join(options.output.split('_'))

    X_train, X_test, one_hots_train, \
    one_hots_test, y_train, y_test, data = load_data(options, return_data=True)
    model_name = [m for m in os.listdir('Results') if 'h5' in m and options.output in m][0]
    model = load_model(os.path.join('Results', model_name))

    print('label' in data.columns)
    x = np.linspace(data[options.output].min(), data[options.output].max(), 100)

    # cluster = pd.read_excel('Results/cluster_member.xlsx')
    # cluster_col = pd.Series(np.zeros(len(data), dtype=int), index=data.index)
    # for col in cluster.columns:
    #     s = cluster[col].isna()
    #     s = cluster[col][~s].astype(int)
    #     cluster_col[s] = col
    try:
        data['Cluster'] = cluster_descriptors()
    except:
        cluster = pd.read_csv('Results/cluster_member.csv')
        data['Cluster'] = cluster['Cluster']

    print('NN Results:')

    # Observed Order
    if options.output == 'relative energy1':
        observed_order = calculate_order(data, 'relative energy1')
        predicted_order = pd.Series(np.zeros(len(data), dtype=int), index=data.index, name='relative energy1')
    # training dataset
    y_predict = model.predict([X_train, one_hots_train])
    if options.output == 'relative energy1':
        predicted_order.loc[y_train.index] = y_predict.reshape(len(y_train))
    print('Train R2: ', r2_score(y_train, y_predict), 'Train MSE: ', mean_squared_error(y_train, y_predict),
          'Train MAE: ', mean_absolute_error(y_train, y_predict))
    residual = y_train - y_predict.reshape(len(y_train))
    fig, ax = plt.subplots(2, figsize=[14.4,  7.3])
    ax[0].scatter(y_train, residual, c='cyan', label='training set')

    # calculate equation for trendline
    z = np.polyfit(y_train, residual, 1)
    p = np.poly1d(z)
    ax[0].plot(x, p(x), 'k')

    residual = pd.DataFrame({'Observed': y_train, 'Predicted':
        y_predict.reshape(len(y_train))}, index=y_train.index)

    residual['Cluster'] = data.loc[residual.index]['Cluster']
    nn_train_r2_scores = residual.groupby('Cluster').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    if options.output == 'relative energy1':
        residual['organic molecule'] = data.loc[residual.index]['organic molecule']
        nn_org_r2_scores_train = residual.groupby('organic molecule').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    # testing dataset
    y_predict = model.predict([X_test, one_hots_test])
    print('Test R2: ', r2_score(y_test, y_predict), 'Test MSE: ', mean_squared_error(y_test, y_predict),
          'Test MAE: ', mean_absolute_error(y_test, y_predict))
    residual = y_test - y_predict.reshape(len(y_test))
    ax[0].scatter(y_test, residual, c='b', label='test set')

    if options.output == 'relative energy1':
        predicted_order.loc[y_test.index] = y_predict.reshape(len(y_test))
        temp_df = pd.concat([predicted_order, data['label']], axis=1)
        predicted_order = calculate_order(temp_df, 'relative energy1')
        cm = pd.crosstab(observed_order['order'], predicted_order['order'])
    # calculate equation for trendline
    z = np.polyfit(y_test, residual, 1)
    p = np.poly1d(z)
    ax[0].plot(x, p(x), 'k--')
    ax[0].text(.01, .01, 'Hirshfeld NN', ha='left', va='bottom', transform=ax[0].transAxes)
    ax[0].legend(loc='upper right')

    residual = pd.DataFrame({'Observed': y_test, 'Predicted':
        y_predict.reshape(len(y_test))}, index=y_test.index)
    residual['Cluster'] = data.loc[residual.index]['Cluster']
    nn_test_r2_scores = residual.groupby('Cluster').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    if options.output == 'relative energy1':
        residual['organic molecule'] = data.loc[residual.index]['organic molecule']
        nn_org_r2_scores_test = residual.groupby('organic molecule').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    # CGCNN Results
    cgcnn_res_dir = 'Results/CGCNN_results'
    train_result = pd.read_csv(os.path.join(cgcnn_res_dir,
                                            'train_results_' + options.output + '.csv'), header=None)
    train_result.columns = ['Id', 'Observed', 'Predicted']
    train_result = train_result.set_index('Id')
    test_result = pd.read_csv(os.path.join(cgcnn_res_dir,
                                           'test_results_' + options.output + '.csv'), header=None)
    test_result.columns = ['Id', 'Observed', 'Predicted']
    test_result = test_result.set_index('Id')

    print('CGCNN Results')
    print('Train R2: ', r2_score(train_result['Observed'], train_result['Predicted']),
          'Train MSE: ', mean_squared_error(train_result['Observed'], train_result['Predicted']),
          'Train MAE: ', mean_absolute_error(train_result['Observed'], train_result['Predicted']))

    print('Test R2: ', r2_score(test_result['Observed'], test_result['Predicted']),
          'Test MSE: ', mean_squared_error(test_result['Observed'], test_result['Predicted']),
          'Test MAE: ', mean_absolute_error(test_result['Observed'], test_result['Predicted']))

    train_result['Cluster'] = data.loc[train_result.index]['Cluster']
    cgcnn_train_r2_scores = train_result.groupby('Cluster').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    test_result['Cluster'] = data.loc[test_result.index]['Cluster']
    cgcnn_test_r2_scores = test_result.groupby('Cluster').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    if options.output == 'relative energy1':
        train_result['organic molecule'] = data.loc[train_result.index]['organic molecule']
        cgcnn_org_r2_scores_test = train_result.groupby('organic molecule').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))
        test_result['organic molecule'] = data.loc[test_result.index]['organic molecule']
        cgcnn_org_r2_scores_train = train_result.groupby('organic molecule').apply(lambda row:
                                                  r2_score(row['Observed'], row['Predicted']))

    ax[1].scatter(train_result['Observed'],
               train_result['Observed'] - train_result['Predicted'], c='pink', label='training set')
    # calculate equation for trendline
    z = np.polyfit(train_result['Observed'], train_result['Observed'] - train_result['Predicted'], 1)
    p = np.poly1d(z)
    ax[1].plot(x, p(x), 'k')

    ax[1].scatter(test_result['Observed'],
               test_result['Observed'] - test_result['Predicted'], c='red', label='test set')
    # calculate equation for trendline
    z = np.polyfit(test_result['Observed'], test_result['Observed'] - test_result['Predicted'], 1)
    p = np.poly1d(z)
    ax[1].plot(x, p(x), 'k--')
    ax[1].text(.01, .01, 'CGCNN', ha='left', va='bottom', transform=ax[1].transAxes)
    ax[1].legend(loc='upper right')

    if options.output == 'relative energy1':
        # ax[0].set_ylim([-0.25, 0.45])
        ax[0].set_ylabel('Residual [ev/atom]')
        # ax[1].set_ylim([-0.25, 0.425])
        ax[1].set_ylabel('Residual [ev/atom]')
        ax[1].set_xlabel('Formation Energy [ev/atom]')
    elif options.output == 'HSE bandgap':
        ax[0].set_ylabel('Residual')
        ax[1].set_ylabel('Residual')
        ax[1].set_xlabel('HSE Band Gap [ev]')
    elif options.output == 'dielectric constant, electronic':
        ax[0].set_ylabel('Residual')
        ax[1].set_ylabel('Residual')
        ax[1].set_xlabel('Electronic Dielectric Constant')

    plt.savefig('Figures/' + options.output + '.png', bbox_inches='tight')
    plt.show()

    r2_scores_table = pd.concat([nn_test_r2_scores, nn_train_r2_scores,
                                 cgcnn_test_r2_scores, cgcnn_train_r2_scores], axis=1)
    r2_scores_table.columns = ['HFS NN test', 'HFS NN train', 'CGCNN test', 'CGCNN train']
    fig, ax = plt.subplots(figsize=[8.65, 7.28])
    sns.heatmap(r2_scores_table, annot=True, ax=ax)
    plt.savefig('Figures/' + options.output + '_cluster.png', bbox_inches='tight')
    plt.show()

    if options.output == 'relative energy1':
        fig, ax = plt.subplots(figsize=[8.65, 7.28])
        sns.heatmap(cm, annot=True, ax=ax, annot_kws={"fontsize":8})
        plt.show()


        r2_org_scores_table = pd.concat([nn_org_r2_scores_train, nn_org_r2_scores_test,
                                     cgcnn_org_r2_scores_train, cgcnn_org_r2_scores_train], axis=1)
        r2_org_scores_table.columns = ['HFS NN test', 'HFS NN train', 'CGCNN test', 'CGCNN train']
        fig, ax = plt.subplots(figsize=[8.65, 7.28])
        sns.heatmap(r2_org_scores_table, annot=True, ax=ax)
        ax.set_title('Formation Energy')
        plt.savefig('Figures/' + options.output + '_organic.png', bbox_inches='tight')
        plt.show()


if __name__ == '__main__':
    main()