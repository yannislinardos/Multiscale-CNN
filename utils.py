import tensorflow as tf
from keras import models
from keras.models import model_from_yaml
import numpy as np
import pickle
import gc
from scipy import linalg
import sympy as sp
from keras.utils import plot_model
import keras


# loads the data from dataframe
def load_data(path):

    # Load Data
    pickle_in = open(path, 'rb')
    data = pickle.load(pickle_in)
    pickle_in.close()

    X = np.array(data['data'])
    Y = np.array(data['class'])

    # Reshape
    X = np.stack(X)
    X = np.expand_dims(X, axis=3)
    Y = tf.keras.utils.to_categorical(Y, 5)

    # Memory
    del data
    gc.collect()

    print('Data loaded from disk')

    return X, Y


# saves model and weights
def save_model(model, name):
    # serialize model to YAML
    model_yaml = model.to_yaml()
    with open("Models/{}.yaml".format(name), "w") as yaml_file:
        yaml_file.write(model_yaml)

    # serialize weights to HDF5
    model.save_weights("Models/{}.h5".format(name))

    print("Saved model to disk")


# loads model from files
def load_model(path_to_model, path_to_weights):

    yaml_file = open(path_to_model, 'r')
    loaded_model_yaml = yaml_file.read()
    yaml_file.close()
    model = model_from_yaml(loaded_model_yaml)

    # load weights into new model
    model.load_weights(path_to_weights)

    print('Model loaded from disk')

    return model


def plot_model(model_name, path_to_plot):

    model = load_model('{}.yaml'.format(model_name), '{}.h5'.format(model_name))

    keras.utils.plot_model(model, to_file='{}.png'.format(path_to_plot), show_shapes=True, show_layer_names=False)


# tests a model and returns score
def test_model(model, X, Y, loss='categorical_crossentropy'):

    model.compile(loss=loss, optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(X, Y, verbose=1, batch_size=1)[1]
    print("%s: %.2f%%" % (model.metrics_names[1], score* 100))

    return score*100


'''
In weights with biases: 0 is weights, 1 is biases

For weights
[] gives previous nodes
[][] denotes dimension of kernel, so we always put 0
[][][][] denotes the weight itself
'''

# the function gets the weights of a 1D convolutional layer and returns a list of the conv kernels (as lists)
# the biases are ignored
def get_kernels(weights: np.ndarray) -> np.ndarray:

    all_kernels = []
    # loop over nodes of current layer
    for current_node in range(len(weights[0][0])):

        current_node_kernels = []

        # loop over nodes of previous layer
        for previous_node in range(len(weights[0])):

            # loop over kernel size
            kernel = []
            for kernel_entry in range(len(weights)):
                weight = weights[kernel_entry][previous_node][current_node]
                kernel.append(weight)

            current_node_kernels.append(kernel)

        all_kernels.append(current_node_kernels)

    return np.array(all_kernels)


# it gets a list of kernels and returns the weights to be put back in the model (not including biases)
def get_weights(kernels: np.ndarray) -> np.ndarray:

    weights = np.ndarray(shape=(tuple(np.flip(kernels.shape))), dtype=np.float32)

    for current_node in range(kernels.shape[0]):

        # loop over nodes of previous layer
        for previous_node in range(kernels.shape[1]):

            # loop over kernel size
            for element_index in range(kernels.shape[2]):
                weights[element_index][previous_node][current_node] = kernels[current_node][previous_node][element_index]

    return weights


# it gets the kernel and the dimension of the desired matrix and it returns the corresponding toeplitz matrix
def get_toeplitz(__kernel: np.ndarray, __matrix_dim: int, zero_padding=True) -> np.ndarray:

    if zero_padding is False:
        return np.transpose(linalg.toeplitz(np.hstack(
            [__kernel, np.zeros(__matrix_dim - len(__kernel))]), np.hstack([__kernel[0], np.zeros(__matrix_dim - 1)])))
    else:
        __extra_size = int(len(__kernel)/2) # for zero padding

        __Toeplitz = linalg.toeplitz(np.hstack(
            [__kernel, np.zeros(__matrix_dim - len(__kernel)+__extra_size)]), np.hstack([__kernel[0], np.zeros(__matrix_dim - 1)]))

        __Toeplitz = np.delete(__Toeplitz, slice(__extra_size), 0)

        return np.transpose(__Toeplitz)

        #return np.transpose(linalg.toeplitz(np.hstack(
            #[__kernel, np.zeros(__matrix_dim - len(__kernel)+__extra_size)]), np.hstack([__kernel[0], np.zeros(__matrix_dim - 1)])))


# inverse of get_toeplitz(), it takes the transpose of toeplitz
def get_kernel_from_toeplitz(__matrix: np.ndarray, kernel_dim: int, __remove_zero_padding=True) -> np.ndarray:

    if __remove_zero_padding is False:
        return np.transpose(__matrix)[:,0][0:kernel_dim]

    # because we get the matrix which is appropriate for zero padding without changing the signal
    else:
        __new_matrix = np.transpose(__matrix)
        __zero_cols = int(kernel_dim / 2)
        __new_matrix = np.delete(__new_matrix, slice(__zero_cols), 1)
        return __new_matrix[:, 0][0:kernel_dim]



def get_symbol_matrix(rows: int, columns: int) -> (sp.Matrix, list):

    __matrix = sp.Matrix.zeros(rows,columns)
    __symbols = []
    for __r in range(rows):
        for __c in range(columns):
            __s = sp.Symbol('X{},{}'.format(__r, __c))
            __symbols.append(__s)
            __matrix[__r, __c] = __s

    return __matrix, __symbols


def get_matrix_from_symbols(rows: int, columns: int, solutions: dict) -> np.ndarray:

    __matrix = np.zeros(shape=(rows, columns))

    for __r in range(rows):
        for __c in range(columns):
            __s = sp.Symbol('X{},{}'.format(__r, __c))
            if __s in solutions.keys():
                __matrix[__r][__c] = solutions[__s]

    return __matrix


def strided_toeplitz(__kernel, __signal_size, __strides=2):

    __rows = __signal_size//2
    __kernel_counter = 0
    __kernel_size = len(__kernel)
    __matrix = np.zeros(shape=(__rows, __signal_size), dtype=object)
    __col = 0
    __prev_col = 0
    for __r in range(__rows):
        __kernel_counter = 0
        while __kernel_counter < __kernel_size and __col < __signal_size:
            __matrix[__r][__col] = __kernel[__kernel_counter%__kernel_size]
            __kernel_counter = __kernel_counter+1
            __col += 1
        __prev_col += __strides
        __col = __prev_col
    return __matrix


# if __name__ == "__main__":

    # model = load_model('Models/Model_24KHz_80%.yaml', 'Models/Model_24KHz_80%.h5')
    #
    # weights0 = model.layers[0].get_weights()[0]
    # weights1 = model.layers[1].get_weights()[0]
    #
    # print(np.array_equal(weights0, get_weights(get_kernels(weights0))))
    #
    # print(np.array_equal(weights1, get_weights(get_kernels(weights1))))


#
#
#     matrix_dim = 8
#     kernel = np.array([1,2,3,4,5])
#     extra_size = int(len(kernel)/2)
#
#     Toeplitz = linalg.toeplitz(np.hstack([kernel, np.zeros(matrix_dim - len(kernel) + extra_size)]), np.hstack([kernel[0], np.zeros(matrix_dim - 1)]))
#
#     Toeplitz = np.delete(Toeplitz, slice(extra_size), 0)

