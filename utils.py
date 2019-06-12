import tensorflow as tf
from keras import models
from keras.models import model_from_yaml
import numpy as np
import pickle
import gc
from scipy import linalg
import sympy as sp


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


# tests a model and returns score
def test_model(model, X, Y):

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    score = model.evaluate(X, Y, verbose=1, batch_size=3)[1]*100
    print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))

    return score


'''
In weights with biases:
[] gives weights for 0 or biases for 1
[][] gives the weights of a node
[][][] denotes dimension of kernel, so we always put 0
[][][][] denotes the weight itself
'''


# the function gete the weights of a 1D convolutional layer and returns a list of the conv kernels (as lists)
# the biases are ignored
def get_kernels(weights) -> np.ndarray:

    # to work even when we do not give the the biases
    if type(weights) is list:
        pass
    elif type(weights) is np.ndarray:
        weights = [weights, 1]

    kernels = []

    # loop over layer size
    for node in range(len(weights[0][0][0])):
        # loop over kernel size
        kernel = []
        for kernel_entry in range(len(weights[0])):
            weight = weights[0][kernel_entry][0][node]
            kernel.append(weight)
        kernels.append(kernel)

    return np.array(kernels)


# it gets a list of kernels and returns the weights to be put back in the model (not including biases)
def get_weights(kernels):

    weights = np.ndarray(shape=(len(kernels[0]), 1, len(kernels)), dtype=np.float32)

    for kernel_index in range(len(kernels)):
        for element_index in range(len(kernels[kernel_index])):
            weights[element_index][0][kernel_index] = kernels[kernel_index][element_index]

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



# if __name__ == "__main__":
#
#
#     matrix_dim = 8
#     kernel = np.array([1,2,3,4,5])
#     extra_size = int(len(kernel)/2)
#
#     Toeplitz = linalg.toeplitz(np.hstack([kernel, np.zeros(matrix_dim - len(kernel) + extra_size)]), np.hstack([kernel[0], np.zeros(matrix_dim - 1)]))
#
#     Toeplitz = np.delete(Toeplitz, slice(extra_size), 0)

