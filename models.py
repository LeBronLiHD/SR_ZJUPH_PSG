from tkinter.messagebox import NO
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM, GlobalAveragePooling2D, BatchNormalization
import tensorflow_addons as tfa
# from tensorflow_addons.layers import CRF
from keras_contrib.layers import CRF
import params
from tensorflow.keras.applications import ResNet152V2, ResNet101V2, EfficientNetV2L, VGG19


def vgg_19():
    base_model = VGG19(weights=None, include_top=False, pooling=None,
                                input_shape=params.INPUT_SHAPE_3)

    # freeze extraction layers
    base_model.trainable = True

    # add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024 , activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(1024 , activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512 , activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(512 , activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(256 , activation='relu')(x)
    predictions = Dense(params.NUM_CLASS, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # confirm unfrozen layers
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def efficient_net_v2l():
    base_model = EfficientNetV2L(weights=None, include_top=False, pooling='max',
                             input_shape=params.INPUT_SHAPE_3) 

    # freeze extraction layers
    base_model.trainable = True

    # add custom top layers
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.6)(x)
    # x = Dense(512, activation='relu')(x)
    # x = Dropout(0.6)(x)
    # x = Dense(256, activation='relu')(x)
    # x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(params.NUM_CLASS, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # x = Dense(1024, kernel_regularizer = regularizers.l2(l=0.020),activity_regularizer=regularizers.l1(0.008),
    #           bias_regularizer=regularizers.l1(0.008), activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(512, kernel_regularizer = regularizers.l2(l=0.020),activity_regularizer=regularizers.l1(0.008),
    #           bias_regularizer=regularizers.l1(0.008), activation='relu')(x)
    # x = Dropout(0.5)(x)
    # x = Dense(256, kernel_regularizer = regularizers.l2(l=0.020),activity_regularizer=regularizers.l1(0.008),
    #           bias_regularizer=regularizers.l1(0.008), activation='relu')(x)
    # x = Dropout(0.3)(x)
    # x = Dense(128, kernel_regularizer = regularizers.l2(l=0.020),activity_regularizer=regularizers.l1(0.008),
    #           bias_regularizer=regularizers.l1(0.008), activation='relu')(x)
    # x = Dropout(0.3)(x)

    # confirm unfrozen layers
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def resnet_101v2():
    base_model = ResNet101V2(weights=None, include_top=False,
                         input_shape=params.INPUT_SHAPE_3)

    # freeze extraction layers
    base_model.trainable = True

    # add custom top layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024 , activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(1024 , activation='relu')(x)
    x = Dropout(0.7)(x)
    x = Dense(512 , activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(512 , activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256 , activation='relu')(x)
    predictions = Dense(params.NUM_CLASS, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # confirm unfrozen layers
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    model.summary()
    return model

def cnn_1():
    nclass = params.NUM_CLASS
    inp = Input(shape=params.INPUT_SHAPE)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.01)(img_1)

    dense_1 = Dropout(rate=0.01)(Dense(64, activation=activations.relu, name="dense_1")(img_1))
    dense_1 = Dropout(rate=0.05)(Dense(64, activation=activations.relu, name="dense_2")(dense_1))
    dense_1 = Dense(nclass, activation=activations.softmax, name="dense_3")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    # opt = optimizers.Adam(0.001)

    # model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def cnn_base():
    inp = Input(shape=params.INPUT_SHAPE)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(16, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(32, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = SpatialDropout1D(rate=0.01)(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(256, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.01)(img_1)

    dense_1 = Dropout(0.01)(Dense(64, activation=activations.relu, name="dense_1")(img_1))

    base_model = models.Model(inputs=inp, outputs=dense_1)
    # opt = optimizers.Adam(0.001)

    # base_model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    # base_model.summary()
    return base_model


def cnn_2():
    nclass = params.NUM_CLASS

    seq_input = Input(shape=params.INPUT_SHAPE_2)
    base_model = cnn_base()
    # for layer in base_model.layers:
    #     layer.trainable = False
    encoded_sequence = TimeDistributed(base_model)(seq_input)
    encoded_sequence = SpatialDropout1D(rate=0.01)(Convolution1D(128,
                                                               kernel_size=3,
                                                               activation="relu",
                                                               padding="same")(encoded_sequence))
    encoded_sequence = Dropout(rate=0.05)(Convolution1D(128,
                                                               kernel_size=3,
                                                               activation="relu",
                                                               padding="same")(encoded_sequence))

    out = Convolution1D(nclass, kernel_size=3, activation="softmax", padding="same")(encoded_sequence)

    model = models.Model(seq_input, out)

    # model.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def cnn_lstm():
    nclass = params.NUM_CLASS

    seq_input = Input(shape=params.INPUT_SHAPE_2)
    base_model = cnn_base()
    # for layer in base_model.layers:
    #     layer.trainable = False
    encoded_sequence = TimeDistributed(base_model)(seq_input)
    encoded_sequence = Bidirectional(LSTM(100, return_sequences=True))(encoded_sequence)
    encoded_sequence = Dropout(rate=0.5)(encoded_sequence)
    encoded_sequence = Bidirectional(LSTM(100, return_sequences=True))(encoded_sequence)
    out = Convolution1D(nclass, kernel_size=1, activation="softmax", padding="same")(encoded_sequence)

    model = models.Model(seq_input, out)

    # model.compile(optimizers.Adam(0.001), losses.sparse_categorical_crossentropy, metrics=['acc'])
    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def cnn_crf():
    nclass = params.NUM_CLASS

    seq_input = Input(shape=params.INPUT_SHAPE_2)
    base_model = cnn_base()
    # for layer in base_model.layers:
    #     layer.trainable = False
    encoded_sequence = TimeDistributed(base_model)(seq_input)
    encoded_sequence = SpatialDropout1D(rate=0.01)(Convolution1D(128,
                                                    kernel_size=3,
                                                    activation="relu",
                                                    padding="same")(encoded_sequence))
    encoded_sequence = Dropout(rate=0.05)(Convolution1D(128,
                                                        kernel_size=3,
                                                        activation="linear",
                                                        padding="same")(encoded_sequence))


    crf = CRF(nclass, sparse_target=True)

    out = crf(encoded_sequence)


    model = models.Model(seq_input, out)

    # model.compile(optimizers.Adam(lr), crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model
