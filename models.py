from tkinter.messagebox import NO
from tensorflow.keras import optimizers, losses, activations, models
from tensorflow.keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate, SpatialDropout1D, TimeDistributed, Bidirectional, LSTM, GlobalAveragePooling2D, BatchNormalization, Conv1D, \
    ReLU, Flatten, MaxPool2D, RNN, StackedRNNCells, RNN, LSTMCell
import tensorflow_addons as tfa
# from tensorflow_addons.layers import CRF
from keras_contrib.layers import CRF
import params
from tensorflow.keras.applications import ResNet152V2, ResNet101V2, EfficientNetV2L, VGG19, VGG16, EfficientNetB7
import tensorflow as tf

def vgg_16():
    base_model = VGG16(weights=None, include_top=False, pooling=None,
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

def efficient_net_b7():
    base_model = EfficientNetB7(weights=None, include_top=False, pooling='max',
                             input_shape=params.INPUT_SHAPE_3) 

    # freeze extraction layers
    base_model.trainable = True

    # add custom top layers
    x = base_model.output
    x = BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001)(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.6)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(params.NUM_CLASS, activation='softmax')(x)
    model = models.Model(inputs=base_model.input, outputs=predictions)

    # confirm unfrozen layers
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def resnet_152v2():
    base_model = ResNet152V2(weights=None, include_top=False,
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
    img_1 = Dropout(rate=0.5)(img_1)

    dense_1 = Dropout(0.1)(Dense(64, activation=activations.relu, name="dense_1")(img_1))

    base_model = models.Model(inputs=inp, outputs=dense_1)
    # opt = optimizers.Adam(0.001)

    # base_model.compile(optimizer=opt, loss=losses.sparse_categorical_crossentropy, metrics=['acc'])
    # base_model.summary()
    return base_model


def cnn_2():
    nclass = params.NUM_CLASS

    seq_input = Input(shape=params.INPUT_SHAPE_2)
    base_model = cnn_base()
    for layer in base_model.layers:
        layer.trainable = True
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
    for layer in base_model.layers:
        layer.trainable = True
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
    for layer in base_model.layers:
        layer.trainable = True
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

#@static
def _tinysleepnet_cnn_2d():
    first_filter_size = int(params.params_tiny["SAMPLE_RATE"] / 2.0)
    first_filter_stride = int(params.params_tiny["SAMPLE_RATE"] / 16.0)
    seq_input = Input(shape=params.INPUT_SHAPE)

    # Weight initializer
    weight_initializer = tf.keras.initializers.VarianceScaling(
        scale=1.0,
        mode="fan_in",
        distribution="normal",
    )
    bias_initializer = tf.zeros_initializer()
    net = Convolution1D(128, 
                        kernel_size=(first_filter_size), 
                        strides=(first_filter_stride), 
                        padding="same",
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer)(seq_input)
    net = BatchNormalization(momentum=0.99, 
                             epsilon=0.001,
                             center=True,
                             scale=True,
                             beta_initializer=tf.zeros_initializer(),
                             gamma_initializer=tf.ones_initializer(),
                             moving_mean_initializer=tf.zeros_initializer(),
                             moving_variance_initializer=tf.ones_initializer())(net)
    net = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(net)
    net = MaxPool1D(pool_size=(8),
                    strides=(8),
                    padding="same")(net)
    net = Dropout(rate=0.5)(net)
    net = Convolution1D(128, 
                        kernel_size=(8), 
                        strides=(1), 
                        padding="same",
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer)(seq_input)
    net = BatchNormalization(momentum=0.99, 
                             epsilon=0.001,
                             center=True,
                             scale=True,
                             beta_initializer=tf.zeros_initializer(),
                             gamma_initializer=tf.ones_initializer(),
                             moving_mean_initializer=tf.zeros_initializer(),
                             moving_variance_initializer=tf.ones_initializer())(net)
    net = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(net)
    net = Convolution1D(128, 
                        kernel_size=(8), 
                        strides=(1), 
                        padding="same",
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer)(seq_input)
    net = BatchNormalization(momentum=0.99, 
                             epsilon=0.001,
                             center=True,
                             scale=True,
                             beta_initializer=tf.zeros_initializer(),
                             gamma_initializer=tf.ones_initializer(),
                             moving_mean_initializer=tf.zeros_initializer(),
                             moving_variance_initializer=tf.ones_initializer())(net)
    net = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(net)
    net = Convolution1D(128, 
                        kernel_size=(8), 
                        strides=(1), 
                        padding="same",
                        kernel_initializer=weight_initializer,
                        bias_initializer=bias_initializer)(seq_input)
    net = BatchNormalization(momentum=0.99, 
                             epsilon=0.001,
                             center=True,
                             scale=True,
                             beta_initializer=tf.zeros_initializer(),
                             gamma_initializer=tf.ones_initializer(),
                             moving_mean_initializer=tf.zeros_initializer(),
                             moving_variance_initializer=tf.ones_initializer())(net)
    net = ReLU(max_value=None, negative_slope=0.0, threshold=0.0)(net)
    net = MaxPool1D(pool_size=(4),
                    strides=(4),
                    padding="same")(net)
    net = GlobalMaxPool1D()(net)
    net = Dropout(rate=0.5)(net)
    net = Dense(64, activation=activations.relu, name="dense_1")(net)
    model = models.Model(seq_input, net)

    # model.compile(optimizers.Adam(lr), crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

#@static
def _tinysleepnet_rnn_2d():

    # input_dim = input_net.shape[-1]
    # print(input_net.shape)
    # seq_input_reshape = tf.reshape(input_net, shape=[-1, params.train_tiny["SEQ_LENGTH"], input_dim], name="reshape_seq_inputs")
    # print(seq_input_reshape.shape)
    representation_model = _tinysleepnet_cnn_2d()
    for layer in representation_model.layers:
        layer.trainable = True
    seq_input = Input(shape=params.INPUT_SHAPE_4)
    encoded_sequence = TimeDistributed(representation_model)(seq_input)
    encoded_sequence = Bidirectional(LSTM(params.params_tiny["RNN_UNITS"], return_sequences=True))(encoded_sequence)
    encoded_sequence = Dropout(rate=0.5)(encoded_sequence)
    encoded_sequence = Bidirectional(LSTM(params.params_tiny["RNN_UNITS"], return_sequences=True))(encoded_sequence)
    out = Convolution1D(params.NUM_CLASS, kernel_size=1, activation="softmax", padding="same")(encoded_sequence)
    model = models.Model(seq_input, out)

    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

    def _create_rnn_cell(n_units):
        """A function to create a new rnn cell."""
        cell = LSTMCell(units=n_units)
        # Dropout wrapper
        keep_prob = tf.cond(params.params_tiny["IS_TRAINING"], lambda:tf.constant(0.5), lambda:tf.constant(1.0))
        cell = tf.nn.RNNCellDropoutWrapper(cell, output_keep_prob=keep_prob)
        return cell

    # LSTM
    cells = []
    for l in range(params.params_tiny["RNN_LAYER"]):
        cells.append(_create_rnn_cell(params.params_tiny["RNN_UNITS"]))

    # Multiple layers of forward and backward cells
    multi_cell = StackedRNNCells(cells=cells)
    # Initial states
    init_state = multi_cell.get_initial_state(batch_size=params.train_tiny["BATCH_SIZE"], 
                                              dtype=tf.float32)
    net = RNN(cell=multi_cell,
              return_sequences=False,
              return_state=False,
              go_backwards=False,
              stateful=False,
              unroll=False,
              time_major=False)(inputs=seq_input_reshape, 
                                training=params.params_tiny["IS_TRAINING"],
                                initial_state=init_state)
    net = tf.reshape(net, shape=[-1, params.params_tiny["RNN_UNITS"]], name="reshape_nonseq_input")

    # seq_input = Input((params.train_tiny["SEQ_LENGTH"], params.params_tiny["INPUT_SIZE"]))
    model = models.Model(seq_input_reshape, net)

    # model.compile(optimizers.Adam(lr), crf.loss_function, metrics=[crf.accuracy])
    model.summary()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model

def tinysleepnet():
    if params.params_tiny['USE_RNN']:
        model = _tinysleepnet_rnn_2d()
    else:
        model = _tinysleepnet_cnn_2d()
    for layer in model.layers:
        if layer.trainable==True:
            print(layer)
    return model
