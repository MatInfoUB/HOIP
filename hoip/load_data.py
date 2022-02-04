from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd


def load_data(options):

    data = pd.read_csv('Data/HOIP_dataframe.csv')
    descriptor_list = pd.read_csv('Data/descriptors_2.txt')
    descriptor_list = descriptor_list['Descriptors']

    # seperating the scalar
    X = data[descriptor_list[:-7]]
    X = StandardScaler().fit_transform(X)
    one_hots = data[descriptor_list[-7:]].astype('float')

    options.input_shape_1 = X.shape[1:]
    options.input_shape_2 = one_hots.shape[1:]

    y = data[options.output]

    # Train test split
    X_train, X_test, one_hots_train, one_hots_test, y_train, y_test = \
        train_test_split(X, one_hots, y, random_state=42)

    return X_train, X_test, one_hots_train, one_hots_test, y_train, y_test