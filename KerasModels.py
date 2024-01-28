"""
A selective collection of CTC-based network architectures to allow easy experimentation.

"""

import keras
import logging
import tensorflow as tf
import keras.backend as K
from typing import Tuple
from keras.models import Model
from keras.layers import (
    Add,
    Activation,
    BatchNormalization,
    Bidirectional,
    Conv1D,
    Conv2D,
    Dense,
    Dropout,
    GlobalAveragePooling1D,
    GRU,
    Input,
    Lambda,
    Layer,
    LSTM,
    MaxPooling2D,
    MaxPool2D,
    Multiply,
    PReLU,
    RandomRotation,
    Reshape,
    ReLU,
    LeakyReLU
)
from keras.constraints import MaxNorm
from keras.optimizers import Adam, RMSprop, Lion


# optional: define a stack of augmentations
# see here for examples: https://www.tensorflow.org/tutorials/images/data_augmentation
#
data_augmentation = keras.Sequential(
    [
        RandomRotation(factor=(0.01, 0.02)),
    ]
)
"""
1) Easter2: https://github.com/kartikgill/Easter2/blob/main/src/easter_model.py
# uses standard values from the reference implementation

"""

BATCH_NORM_EPSILON = 1e-5
BATCH_NORM_DECAY = 0.997


def ctc_loss(args):
    y_pred, labels, input_length, label_length = args
    return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


def ctc_custom(args):
    """
    custom CTC loss, see Easter2 paper
    """
    y_pred, labels, input_length, label_length = args
    ctc_loss = K.ctc_batch_cost(labels, y_pred, input_length, label_length)
    p = tf.exp(-ctc_loss)
    gamma = 0.5
    alpha = 0.25
    return alpha * (K.pow((1 - p), gamma)) * ctc_loss


def batch_norm(inputs):
    return BatchNormalization(momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON)(
        inputs
    )


def add_global_context(data, filters):
    """
    1D Squeeze and Excitation Layer.
    """
    pool = GlobalAveragePooling1D()(data)

    pool = Dense(filters // 8, activation="relu")(pool)

    pool = Dense(filters, activation="sigmoid")(pool)

    final = Multiply()([data, pool])
    return final


def easter_unit(old, data, filters, kernel, stride, dropouts):
    """
    Easter unit with dense residual connections
    """
    old = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(old)
    old = batch_norm(old)

    this = Conv1D(filters=filters, kernel_size=1, strides=1, padding="same")(data)
    this = batch_norm(this)

    old = Add()([old, this])

    data = Conv1D(filters=filters, kernel_size=kernel, strides=stride, padding="same")(
        data
    )

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(dropouts)(data)

    data = Conv1D(filters=filters, kernel_size=kernel, strides=stride, padding="same")(
        data
    )

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(dropouts)(data)

    data = Conv1D(filters=filters, kernel_size=kernel, strides=stride, padding="same")(
        data
    )

    data = batch_norm(data)
    data = add_global_context(data, filters)

    final = Add()([old, data])

    data = Activation("relu")(final)
    data = Dropout(dropouts)(data)

    return data, old


def Easter2(
    input_width: int = 2000,
    input_height: int = 80,
    classes: int = 68,
    learning_rate: float = 0.001,
    optimizer: str = "Adam",
    output_dim: int = 500,  # standard value in paper, experiment with other values if necessary
):
    inputs = Input(name="images", shape=(input_width, input_height))

    # data = data_augmentation(inputs) # optional: use augmentation layers, some keras layers cause problems though
    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(inputs)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    old = data

    # 3 * 3 Easter Blocks (with dense residuals)
    data, old = easter_unit(old, data, 128, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2)
    data, old = easter_unit(old, data, 256, 9, 1, 0.3)
    #data, old = easter_unit(old, data, 512, 14, 1, 0.3)
    """The following three units are added for the latest test run"""
    #data, old = easter_unit(old, data, 512, 9, 1, 0.3)
    #data, old = easter_unit(old, data, 512, 11, 1, 0.3)
    #data, old = easter_unit(old, data, 512, 13, 1, 0.3)

    
    data = Conv1D(
        filters=512, kernel_size=11, strides=1, padding="same", dilation_rate=2
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=512, kernel_size=1, strides=1, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)
    #data = Dense(units=output_dim, activation="relu")(data)

    data = Conv1D(filters=classes, kernel_size=1, strides=1, padding="same")(data)

    y_pred = Activation("softmax", name="Final")(data)
    print(f"Y_pred: {y_pred}")

    """
       define an optimizer, check the docs for optional parameters
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop
       """

    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        optimizer = RMSprop(learning_rate=learning_rate, centered=True)
    elif optimizer == "Lion":
        optimizer = Lion(learning_rate=learning_rate)
    else:
        logging.info(f"setting optimzer to default: Adam")
        optimizer = Adam(learning_rate=learning_rate)

    labels = Input(name="labels", shape=[output_dim], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    output = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    # compiling model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=output)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    return model


def Easter2a(
    input_width: int = 2560,
    input_height: int = 80,
    classes: int = 68,
    learning_rate: float = 0.001,
    optimizer: str = "Adam",
    output_dim: int = 500,  # standard value in paper, experiment with other values if necessary
):
    inputs = Input(name="images", shape=(input_width, input_height))

    # data = data_augmentation(inputs) # optional: use augmentation layers, some keras layers cause problems though
    data = Conv1D(filters=512, kernel_size=3, strides=2, padding="same")(inputs)
    data = batch_norm(data)
    data = Activation("tanh")(data)
    data = Dropout(0.2)(data)
    #print(f"Conv1D-512: {data.shape}")

    data = Conv1D(filters=256, kernel_size=3, strides=1, padding="same")(data)
    data = batch_norm(data)
    data = Activation("tanh")(data)
    data = Dropout(0.2)(data)
    #print(f"Conv1D-256: {data.shape}")

    data = Conv1D(filters=256, kernel_size=3, strides=2, padding="same")(data)
    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)
    p#rint(f"Conv1D-256: {data.shape}")

    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(data)
    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)
    #print(f"Conv1D-128: {data.shape}")

    old = data

    # 3 * 3 Easter Blocks (with dense residuals)
    data, old = easter_unit(old, data, 128, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2)
    data, old = easter_unit(old, data, 512, 14, 1, 0.3)
    #data, old = easter_unit(old, data, 512, 11, 1, 0.3)

    data = Conv1D(
        filters=1024, kernel_size=11, strides=1, padding="same", dilation_rate=2
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    ## just added another Conv1D
    data = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(data)
    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)
    

    data = Conv1D(filters=classes, kernel_size=1, strides=1, padding="same")(data)
    #print(f"After Last Conv1D: {data.shape}")
    y_pred = Activation("softmax", name="Final")(data)
    #print(f"y_pred: {y_pred.shape}")

    """
       define an optimizer, check the docs for optional parameters
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop
       - TODO: parameterize optimizer parameters, add Lion optimizer for tf > 2.12?
       """

    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        optimizer = RMSprop(learning_rate=learning_rate, centered=True)
    elif optimizer == "Lion":
        optimizer = Lion(learning_rate=learning_rate)
    else:
        logging.info(f"setting optimzer to default: Adam")
        optimizer = Adam(learning_rate=learning_rate)

    labels = Input(name="labels", shape=[output_dim], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    output = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    # compiling model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=output)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    logging.info(model.summary)

    return model



def Easter2b(
    input_width: int = 2000,
    input_height: int = 80,
    classes: int = 68,
    learning_rate: float = 0.001,
    optimizer: str = "Adam",
    output_dim: int = 500,  # standard value in paper, experiment with other values if necessary
):
    inputs = Input(name="images", shape=(input_width, input_height))

    # data = data_augmentation(inputs) # optional: use augmentation layers, some keras layers cause problems though
    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(inputs)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    data = Conv1D(filters=128, kernel_size=3, strides=2, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.2)(data)

    old = data

    # 3 * 3 Easter Blocks (with dense residuals)
    data, old = easter_unit(old, data, 128, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 5, 1, 0.2)
    data, old = easter_unit(old, data, 256, 7, 1, 0.2)
    data, old = easter_unit(old, data, 256, 9, 1, 0.3)
    #data, old = easter_unit(old, data, 512, 14, 1, 0.3)
    """The following three units are added for the latest test run"""
    data, old = easter_unit(old, data, 512, 9, 1, 0.3)
    data, old = easter_unit(old, data, 512, 11, 1, 0.3)
    data, old = easter_unit(old, data, 512, 13, 1, 0.3)

    
    data = Conv1D(
        filters=1024, kernel_size=11, strides=1, padding="same", dilation_rate=2
    )(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)

    data = Conv1D(filters=1024, kernel_size=1, strides=1, padding="same")(data)

    data = batch_norm(data)
    data = Activation("relu")(data)
    data = Dropout(0.4)(data)
    #data = Dense(units=output_dim, activation="relu")(data)

    data = Conv1D(filters=classes, kernel_size=1, strides=1, padding="same")(data)

    y_pred = Activation("softmax", name="Final")(data)
    print(f"Y_pred: {y_pred}")

    """
       define an optimizer, check the docs for optional parameters
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/Adam
       - https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/RMSprop
       """

    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        optimizer = RMSprop(learning_rate=learning_rate, centered=True)
    elif optimizer == "Lion":
        optimizer = Lion(learning_rate=learning_rate)
    else:
        logging.info(f"setting optimzer to default: Adam")
        optimizer = Adam(learning_rate=learning_rate)

    labels = Input(name="labels", shape=[output_dim], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    output = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )

    # compiling model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=output)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    logging.info(model.summary)
    return model














"""
2. A slightly adapted version from Arthur Flor's Flor++ Network using FullyGatedConv2D-Layers
- https://github.com/arthurflor23/handwritten-text-recognition/blob/master/src/network/model.py
"""

class FullGatedConv2D(Conv2D):
    """Gated Convolutional Class"""

    def __init__(self, filters, **kwargs):
        super(FullGatedConv2D, self).__init__(filters=filters * 2, **kwargs)
        self.nb_filters = filters

    def call(self, inputs):
        """Apply gated convolution"""

        output = super(FullGatedConv2D, self).call(inputs)
        linear = Activation("linear")(output[:, :, :, :self.nb_filters])
        sigmoid = Activation("sigmoid")(output[:, :, :, self.nb_filters:])

        return Multiply()([linear, sigmoid])

    def compute_output_shape(self, input_shape):
        """Compute shape of layer output"""

        output_shape = super(FullGatedConv2D, self).compute_output_shape(input_shape)
        return tuple(output_shape[:3]) + (self.nb_filters * 2,)

    def get_config(self):
        """Return the config of the layer"""

        config = super(FullGatedConv2D, self).get_config()
        config['nb_filters'] = self.nb_filters
        del config['filters']
        return config



def FlorCRNN(
    input_width: int = 2000,
    input_height: int = 80,
    classes: int = 68,
    learning_rate: float = 0.001,
    optimizer: str = "Adam",
    output_dim: int = 500,
    kernel_initializer = "he_uniform"
):

    input_size = (input_height, input_width, 1)

    inputs = Input(name="images", shape=input_size)

    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(2, 2), padding="same", kernel_initializer=kernel_initializer)(inputs)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    x = FullGatedConv2D(filters=16, kernel_size=(3, 3), padding="same")(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    x = FullGatedConv2D(filters=32, kernel_size=(3, 3), padding="same")(x)
    x = Conv2D(filters=40, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer=kernel_initializer)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    x = FullGatedConv2D(filters=40, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    x = FullGatedConv2D(filters=48, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(x)
    x = Dropout(rate=0.2)(x)

    x = Conv2D(filters=56, kernel_size=(2, 4), strides=(2, 4), padding="same", kernel_initializer=kernel_initializer)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    x = FullGatedConv2D(filters=56, kernel_size=(3, 3), padding="same", kernel_constraint=MaxNorm(4, [0, 1, 2]))(x)
    x = Dropout(rate=0.2)(x)
    x = Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    x = Conv2D(filters=64, kernel_size=(2, 2), strides=(1, 1), padding="same", kernel_initializer=kernel_initializer)(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = BatchNormalization(renorm=True)(x)
    #cnn = MaxPooling2D(pool_size=(1, 2), strides=(1, 2), padding="valid")(cnn)
    shape = x.get_shape()
    print(f"X-shape: {x.shape}")
    x = Reshape((x.shape[1] * x.shape[2], shape[3]))(x)
    print(f"Reshape-shape: {x.shape}")
    rnn_units = x.shape[1]
    print(f"rnn-units: {rnn_units}")
    x = Bidirectional(GRU(units=rnn_units, return_sequences=True, dropout=0.5))(x)
    x = Dense(units=512)(x)
    y_pred = Dense(classes, activation='softmax')(x)

    labels = Input(name='the_labels', shape=[output_dim], dtype='float32')
    input_length = Input(name='input_length', shape=[1], dtype='int64')
    label_length = Input(name='label_length', shape=[1], dtype='int64')


    def ctc_lambda_func(args):
        y_pred, labels, input_length, label_length = args
        return K.ctc_batch_cost(labels, y_pred, input_length, label_length)


    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        optimizer = RMSprop(learning_rate=learning_rate, centered=True)
    elif optimizer == "Lion":
        optimizer = Lion(learning_rate=learning_rate)
    else:
        logging.info(f"setting optimzer to default: Adam")
        optimizer = Adam(learning_rate=learning_rate)


    labels = Input(name="labels", shape=[output_dim], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    loss_out = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )
    # compiling model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    return model



"""
3. A Vanilla CRNN Network

"""

class ConvReluBlock(Layer):
    """Gated Convolutional Class"""

    def __init__(self, filters: int, kernel: int = 3, strides: int = 1, use_bn: bool = False, dropout: float = 0.2, leaky_relu: bool = False, padding: str = "valid"):
        super(ConvReluBlock, self).__init__()
        self.filters = filters
        self.kernel = kernel
        self.strides = strides
        self.padding = padding
        self.use_bn = use_bn
        self.dropout = dropout
        self.leaky_relu = leaky_relu

    def call(self, inputs):
        print(f"Calling ConvReluBlock: {inputs.shape}")
        x = Conv2D(self.filters, kernel_size=(self.kernel, self.kernel), strides=(self.strides, self.strides), padding=self.padding)(inputs)
        
        if self.use_bn:
            x = BatchNormalization(renorm=True)(x)
        if self.leaky_relu:
            x = LeakyReLU(alpha=0.3)(x)
        else:
            x = ReLU()(x)
        
        x = Dropout(self.dropout)(x)
        return x



def VanillarCRNN(
    input_width: int = 2000,
    input_height: int = 80,
    classes: int = 68,
    learning_rate: float = 0.001,
    optimizer: str = "Adam",
    output_dim: int = 500,
    map2rnn_units = 64,
    rnn_units = 256

):

    input_size = (input_height, input_width, 1)
    inputs = Input(name="images", shape=input_size)

    x = ConvReluBlock(filters=64)(inputs)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = ConvReluBlock(filters=128)(x)
    x = MaxPooling2D(pool_size=2, strides=2)(x)
    x = ConvReluBlock(filters=256)(x)
    x = ConvReluBlock(filters=256)(x)
    x = MaxPooling2D(pool_size=(2,1), strides=(2,1))(x)
    x = ConvReluBlock(filters=512, use_bn=True)(x)
    x = ConvReluBlock(filters=512, use_bn=True)(x)
    x = MaxPooling2D(pool_size=(2,1), strides=(2,1))(x)
    print(f"Current shape 1: {x.shape}")
    x = ConvReluBlock(filters=512, kernel=2, padding="valid")(x)
    #x = Dense(map2rnn_units)
    print(f"Current shape after last ConvReluBlock: {x.shape}")
    shape = x.get_shape()
    x = Reshape((shape[1] * shape[2], shape[3]))(x)
    print(f"Reshaped shape: {x.shape}")
    x = Bidirectional(LSTM(rnn_units, return_sequences=True))(x)
    x = Bidirectional(LSTM(2*rnn_units, return_sequences=True))(x)
    y_pred = Dense(classes, activation='softmax')(x)

    if optimizer == "Adam":
        optimizer = Adam(learning_rate=learning_rate)
    elif optimizer == "RMSProp":
        optimizer = RMSprop(learning_rate=learning_rate, centered=True)
    elif optimizer == "Lion":
        optimizer = Lion(learning_rate=learning_rate)
    else:
        logging.info(f"setting optimzer to default: Adam")
        optimizer = Adam(learning_rate=learning_rate)


    labels = Input(name="labels", shape=[output_dim], dtype="float32")
    input_length = Input(name="input_length", shape=[1], dtype="int64")
    label_length = Input(name="label_length", shape=[1], dtype="int64")

    loss_out = Lambda(ctc_custom, output_shape=(1,), name="ctc")(
        [y_pred, labels, input_length, label_length]
    )
    # compiling model
    model = Model(inputs=[inputs, labels, input_length, label_length], outputs=loss_out)
    model.compile(loss={"ctc": lambda y_true, y_pred: y_pred}, optimizer=optimizer)

    return model



"""
4. Attention-based Model
"""
class Encoder:

    layers = [
        Conv2D(64, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

        Conv2D(128, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        MaxPool2D(pool_size=(2, 2), strides=(2, 2), padding='valid'),

        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Conv2D(256, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='he_uniform'),
        MaxPool2D(pool_size=(2, 1), strides=(2, 1), padding='valid'),

        Conv2D(512, (2, 2), padding='valid', activation='relu', kernel_initializer='he_uniform'),
        BatchNormalization(),
        Dropout(rate=0.5)
    ]

    def __init__(self, units):
        assert units % 2 == 0  # units must be even, because the encoder is bidirectional
        self.cnn = Sequential(self.layers)
        self.lstm = Bidirectional(LSTM(units // 2, return_sequences=True))
        self.cnn_shape = None

    def __call__(self, encoder_input) -> tf.Tensor:
        out = self.cnn(encoder_input)
        out = tf.squeeze(out, axis=1)
        self.cnn_shape = out.shape
        lstm = self.lstm(out)
        return lstm

    @staticmethod
    def get_width(width):
        for layer in Encoder.layers:
            if type(layer) is MaxPool2D:
                assert layer.strides == layer.pool_size
                width = math.ceil(width / layer.pool_size[1])
            elif type(layer) is Conv2D and layer.padding == 'valid':
                assert layer.strides == (1, 1)
                width = width - math.ceil(layer.kernel_size[1] / 2.0)
        return width

class Attention:
    def __init__(self, units: int):
        self.query_projection = Dense(units)

    def __call__(self, decoder_input, encoder_output) -> Tuple[tf.Tensor, tf.Tensor]:
        query = self.query_projection(decoder_input)
        key = encoder_output
        value = encoder_output
        logits = tf.matmul(query, tf.transpose(key, perm=[0, 2, 1]))
        attention_weights = tf.nn.softmax(logits)
        context_vectors = tf.matmul(attention_weights, value)
        return [context_vectors, attention_weights]


class Decoder:
    def __init__(self, units):
        self.lstm = LSTM(units, return_sequences=True, return_state=True)

    def __call__(self, decoder_input, initial_state) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        return self.lstm(decoder_input, initial_state)


class DecoderOutput:
    def __init__(self, vocab_size):
        self.hidden = Dense(256, activation='relu')
        self.dense = Dense(vocab_size, activation='softmax')

    def __call__(self, decoder_output) -> tf.Tensor:
        net = self.hidden(decoder_output)
        return self.dense(net)
