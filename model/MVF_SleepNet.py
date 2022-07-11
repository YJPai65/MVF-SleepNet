from keras.models import Model, Sequential, load_model
from model.GLGCN import *
from keras.layers.recurrent import GRU
from keras.layers import Input, Dense, Dropout, Concatenate
from model.Utils import *


def build_MVFSleepNet(cheb_k, num_of_chev_filters, num_of_time_filters, time_conv_strides,
                      time_conv_kernel, sample_shape1, sample_shape2, num_block, dropout,
                      opt, GLalpha):
    featureNet_adjacent_input = Input(shape=sample_shape1)
    Spatial_temporal_feature = Extract_spatial_temporal(k=cheb_k, num_of_chev_filters=num_of_chev_filters,
                                                        num_of_time_filters=num_of_time_filters,
                                                        time_conv_strides=time_conv_strides,
                                                        time_conv_kernel=time_conv_kernel,
                                                        data_layer_input=featureNet_adjacent_input,
                                                        num_block=num_block,
                                                        GLalpha=GLalpha
                                                        )

    STFT_adjacent_input = Input(shape=sample_shape2)
    Spectral_temporal_feature = Extract_spectral_temporal(STFT_adjacent_input)

    merged = Concatenate(axis=1)([Spatial_temporal_feature, Spectral_temporal_feature])
    merged = Dense(256, activation="relu")(merged)
    merged = Dropout(dropout)(merged)
    merged = Dense(5, activation="softmax")(merged)
    model = Model(inputs=[featureNet_adjacent_input, STFT_adjacent_input], outputs=merged)

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['acc'],
    )

    return model


def Extract_spectral_temporal(STFT_adjacent_input):
    x = STFT_adjacent_input
    x_gru = GRU(256, return_sequences=True)(x)
    x_gru = Dropout(0.5)(x_gru)
    x_gru = GRU_Attention(5)(x_gru)
    merged = x_gru
    merged = Dense(256, activation="relu")(merged)

    return merged
