import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

# Loading dataset
data = pd.read_csv('Data/features_dataframe_for_HOIPs_one_hot.csv')
# Loading the set of descriptors
descriptor_list = pd.read_csv('Data/descriptors.txt')
descriptor_list = descriptor_list['Descriptors']

# seperating the scalar
X = data[descriptor_list[:-7]]
X = StandardScaler().fit_transform(X)
one_hots = data[descriptor_list[-7:]].astype('float')

input_shape_1 = X.shape[1:]
input_shape_2 = one_hots.shape[1:]

output_list = pd.read_csv('Data/output_list.csv', sep='\t')
output_list = output_list['Outputs']
output_ind = 0
y = data[output_list[output_ind]]

# Train test split
X_train, X_test, one_hots_train, one_hots_test, y_train, y_test = \
    train_test_split(X, one_hots, y)

from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model

from sklearn.metrics import r2_score

input_scalar = Input(shape=input_shape_1)
input_one_hot = Input(shape=input_shape_2)

one_hot_fea = Dense(input_shape_2[0], activation='sigmoid')(input_one_hot)
pooled_fea = Concatenate(axis=1)([input_scalar, one_hot_fea])
dense1 = Dense(128, activation='relu')(pooled_fea)

output = Dense(1)(dense1)

model = Model(inputs=[input_scalar, input_one_hot], outputs=output)
model.compile(optimizer='Adam', loss='mse')

model.fit(x=[X_train, one_hots_train], y=y_train, epochs=100,
          validation_data=([X_test, one_hots_test], y_test))

y_predict = model.predict([X_test, one_hots_test])
print('Testing R2 is: ', r2_score(y_test, y_predict))

split = np.array(['train'] * len(y))
split[y_test.index] = 'test'
output = pd.DataFrame({output_list[output_ind]: y.values, 'split': split,
                       'Predicted': model.predict([X, one_hots]).reshape(len(y))})
from datetime import datetime
today = datetime.today().strftime('%m_%d_%Y')

# Saving the results
import os
result_folder = os.path.join('Results', output_list[output_ind])
output.to_csv(os.path.join(result_folder, 'Regression_' + today + '.csv')) # Saving the values
model.save(os.path.join(result_folder, 'model_'+today+'.h5')) # saving the model