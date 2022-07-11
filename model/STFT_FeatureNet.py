from keras.models import Model
from keras import layers
from keras.layers import Input, Dropout



def vgg_16_modified(input_shape, classes):
    input_signal = Input(shape=input_shape)
    x = layers.BatchNormalization()(input_signal)

    # block 1
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(64, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 2
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(128, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 3
    x = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 4
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # block 5
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.Conv2D(512, kernel_size=3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(x)

    # FC
    x = layers.Flatten()(x)
    x = layers.Dense(2560)(x)
    x = Dropout(0.5)(x)
    STFT_feature = x
    x = layers.Dense(256)(x)
    x = layers.Dense(5, activation="softmax")(x)

    pred_model = Model(inputs=input_signal, outputs=x)
    fea_model = Model(inputs=input_signal, outputs=STFT_feature)

    return fea_model, pred_model


def build_STFTNet(input_shape, opt):
    # VGGNet
    fea_model, pre_model = vgg_16_modified(input_shape, 5)
    fea_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    # VGGNet without softmax
    pre_model.compile(optimizer=opt,
                      loss='categorical_crossentropy',
                      metrics=['acc'])

    return fea_model, pre_model
