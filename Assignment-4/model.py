from keras.layers import Conv2D, MaxPooling2D, Input
from keras.models import Model
from keras.layers import Dense, Flatten, Dropout
from utils import input_shape
from utils import f1


def get_model(

):
    img_input = Input(name='img_input', shape=input_shape, dtype='float32')

    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                    strides=2, input_shape=input_shape,
                    name='conv_1')(img_input)

    max_pool_1 = MaxPooling2D((2, 2), strides=2, name='pool_1')(conv_1)

    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                    strides=2, name='conv_2')(max_pool_1)

    max_pool_2 = MaxPooling2D((2, 2), strides=2, name='pool_2')(conv_2)

    flatten = Flatten()(max_pool_2)

    dense = Dense(2048, activation='relu', name='dense')(flatten)

    y_pred = Dense(1, activation='sigmoid', name='y_pred')(dense)

    model = Model(input=img_input, output=y_pred)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=[f1, 'accuracy'])

    return model


def get_model_exp(

):
    img_input = Input(name='img_input', shape=input_shape, dtype='float32')

    conv_1 = Conv2D(filters=32, kernel_size=(3, 3), activation='relu',
                    strides=2, input_shape=input_shape,
                    name='conv_1')(img_input)

    max_pool_1 = MaxPooling2D((2, 2), strides=2, name='pool_1')(conv_1)

    conv_2 = Conv2D(filters=64, kernel_size=(3, 3), activation='relu',
                    strides=2, name='conv_2')(max_pool_1)

    max_pool_2 = MaxPooling2D((2, 2), strides=2, name='pool_2')(conv_2)

    dropout_1 = Dropout(0.1)(max_pool_2)

    flatten = Flatten()(dropout_1)

    dense = Dense(2048, activation='relu', name='dense')(flatten)

    dropout_2 = Dropout(0.1)(dense)

    y_pred = Dense(1, activation='sigmoid', name='y_pred')(dropout_2)

    model = Model(input=img_input, output=y_pred)

    model.compile(loss='binary_crossentropy',
                  optimizer='adam', metrics=[f1, 'accuracy'])

    return model
