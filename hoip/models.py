from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input, Concatenate
from tensorflow.keras.models import Model


class RegressionModel:

    def __init__(self, opt):

        self.opt = opt
        self.n_att_layer = opt.n_att_layer
        self.n_fc = opt.n_fc
        self.input_shape_1 = opt.input_shape_1
        self.input_shape_2 = opt.input_shape_2
        self.nn = None

        # Building the Model
        self.build_model()

    def build_model(self):

        input_scalar = Input(shape=self.input_shape_1)
        input_one_hot = Input(shape=self.input_shape_2)

        one_hot_fea = input_one_hot
        for i in range(self.n_att_layer):
            one_hot_fea = Dense(self.input_shape_1[0], activation='relu')(one_hot_fea)

        pooled_fea = Concatenate(axis=1)([input_scalar, one_hot_fea])

        for i in range(self.n_fc):
            pooled_fea = Dense(128, activation='relu')(pooled_fea)

        output = Dense(1)(pooled_fea)
        self.nn = Model(inputs=[input_scalar, input_one_hot], outputs=output)

    def compile(self):

        self.nn.compile(optimizer=self.opt.optimizer, loss=self.opt.loss)

