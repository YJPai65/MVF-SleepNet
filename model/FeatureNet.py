import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Dropout, MaxPool1D, Activation
from keras.layers import Flatten, Reshape, TimeDistributed, BatchNormalization

'''
A Feature Extractor Network
'''

def build_FeatureNet(opt, channels=10, time_second=30, freq=100):
    activation = tf.nn.relu
    padding = 'same'

    ######### Input ########
    input_signal = Input(shape=(time_second * freq, 1), name='input_signal')

    ######### CNNs with small filter size at the first layer #########
    cnn0 = Conv1D(kernel_size=50,
                  filters=32,
                  strides=6,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    s = cnn0(input_signal)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn1 = MaxPool1D(pool_size=16, strides=16)
    s = cnn1(s)
    cnn2 = Dropout(0.5)
    s = cnn2(s)
    cnn3 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn3(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn4 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn4(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn5 = Conv1D(kernel_size=8, filters=64, strides=1, padding=padding)
    s = cnn5(s)
    s = BatchNormalization()(s)
    s = Activation(activation=activation)(s)
    cnn6 = MaxPool1D(pool_size=8, strides=8)
    s = cnn6(s)
    cnn7 = Reshape((int(s.shape[1]) * int(s.shape[2]), ))  # Flatten
    s = cnn7(s)

    ######### CNNs with large filter size at the first layer #########
    cnn8 = Conv1D(kernel_size=400,
                  filters=64,
                  strides=50,
                  kernel_regularizer=keras.regularizers.l2(0.001))
    l = cnn8(input_signal)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn9 = MaxPool1D(pool_size=8, strides=8)
    l = cnn9(l)
    cnn10 = Dropout(0.5)
    l = cnn10(l)
    cnn11 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn11(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn12 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn12(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn13 = Conv1D(kernel_size=6, filters=64, strides=1, padding=padding)
    l = cnn13(l)
    l = BatchNormalization()(l)
    l = Activation(activation=activation)(l)
    cnn14 = MaxPool1D(pool_size=4, strides=4)
    l = cnn14(l)
    cnn15 = Reshape((int(l.shape[1]) * int(l.shape[2]), ))
    l = cnn15(l)

    feature = keras.layers.concatenate([s, l])

    fea_part = Model(input_signal, feature)

    ##################################################

    input = Input(shape=(channels, time_second * freq), name='input_signal')
    reshape = Reshape((channels, time_second * freq, 1))  # Flatten
    input_re = reshape(input)
    fea_all = TimeDistributed(fea_part)(input_re)

    merged = Flatten()(fea_all)
    merged = Dropout(0.5)(merged)
    merged = Dense(64)(merged)
    merged = Dense(5)(merged)

    fea_softmax = Activation(activation='softmax')(merged)

    # FeatureNet with softmax
    fea_model = Model(input, fea_softmax)
    fea_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    # FeatureNet without softmax
    pre_model = Model(input, fea_all)
    pre_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    return fea_model, pre_model
