from keras.models import Sequential
from keras.layers.convolutional import Convolution1D
from keras.layers import Lambda
from keras import backend as K
from keras.engine import InputLayer
import keras
import utils
import numpy as np

def to_fully_conv(model):

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, None, 3), name="input_new")

    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape

        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim = layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],f_dim[3],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution1D(output_dim,
                                          (f_dim[1],f_dim[2]),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution1D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer

        new_model.add(new_layer)

    return new_model


if __name__ == '__main__':

    model = utils.load_model('Models/Model_24KHz_87%_meanpooling.yaml', 'Models/Model_24KHz_87%_meanpooling.h5')

    new_model = Sequential()

    input_layer = InputLayer(input_shape=(None, 1), name="input_new")

    new_model.add(input_layer)

    for layer in model.layers:

        if "Flatten" in str(layer):
            flattened_ipt = True
            f_dim = layer.input_shape


        elif "Dense" in str(layer):

            input_shape = layer.input_shape
            output_dim = layer.get_weights()[1].shape[0]
            W,b = layer.get_weights()

            if flattened_ipt:
                shape = (f_dim[1],f_dim[2],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution1D(output_dim,
                                          f_dim[1],
                                          strides=1,
                                          activation='linear',               #layer.activation,
                                          padding='valid',
                                          weights=[new_W,b], name='converted_conv')
                flattened_ipt = False

            else:
                shape = (1,1,input_shape[1],output_dim)
                new_W = W.reshape(shape)
                new_layer = Convolution1D(output_dim,
                                          (1,1),
                                          strides=(1,1),
                                          activation=layer.activation,
                                          padding='valid',
                                          weights=[new_W,b])


        else:
            new_layer = layer

        new_model.add(new_layer)

    new_model.add(Lambda(lambda x: K.batch_flatten(x)))



    X, Y = utils.load_data('Dataframes/Testing24.pickle')
    score = utils.test_model(new_model, X, Y)
    print("%s: %.2f%%" % (new_model.metrics_names[1], score))
    # np.transpose(np.expand_dims(Y, axis=-1))