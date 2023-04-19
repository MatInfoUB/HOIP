from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


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
    ind_train, ind_test = train_test_split(range(len(data)), random_state=42, train_size=options.train_ratio)
    X_train, X_test = X[ind_train], X[ind_test]
    ind_train, ind_test = train_test_split(data['file'].tolist(), random_state=42)
    y_train, y_test, one_hots_train, one_hots_test= y[ind_train], y[ind_test], one_hots.loc[ind_train], one_hots.loc[ind_test]

    if return_data:
        return X_train, X_test, one_hots_train, one_hots_test, y_train, y_test, data

    else:
        return X_train, X_test, one_hots_train, one_hots_test, y_train, y_test


def calculate_order(df, col=None):

    if 'file' in df.columns:
        df = df.drop(columns='file')
    order_full = []
    file_list = []
    for unique_label in df.label.unique():
        temp = df[df['label'] == unique_label].sort_values(col).reset_index()
        file_list += temp['file'].tolist()
        order_full += temp.index.tolist()

    order = pd.DataFrame({'file': file_list, 'order': order_full})
    order = order.sort_values(by='file')

    return order