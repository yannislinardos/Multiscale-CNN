import utils
import keras
from keras import models
from keras import layers
import numpy as np
import sympy as sp
from scipy.sparse.linalg import lsqr
from scipy import sparse
import scipy


def upscale(method: str, old_model_name: str, new_model_name: str):

    old_model = utils.load_model('Models/{}.yaml'.format(old_model_name), 'Models/{}.h5'.format(old_model_name))

    new_model = models.Sequential()

    first_layer = True
    for layer in old_model.layers:

        if type(layer) is keras.layers.convolutional.Conv1D:

            biases = layer.get_weights()[1]
            old_kernels = utils.get_kernels(layer.get_weights()[0])
            nodes = layer.kernel.shape[2].value

            if method == 'nearest_neighbor':

                new_kernels = nearest_neighbor(old_kernels)
                new_weights = [utils.get_weights(new_kernels), biases]

                if first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                          input_shape=(4*24000, 1), padding='same', weights=new_weights)
                    first_layer = False

                elif not first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                          padding='same', weights=new_weights)

                new_model.add(new_layer)

            elif method == 'linear':

                new_kernels = linear(old_kernels)

                new_weights = [utils.get_weights(new_kernels), biases]

                if first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              input_shape=(4*24000, 1), padding='same', weights=new_weights)
                    first_layer = False

                elif not first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              padding='same', weights=new_weights)

                new_model.add(new_layer)

            elif method == 'distance_weighting':

                new_kernels = distance_weighting(old_kernels)
                new_weights = [utils.get_weights(new_kernels), biases]

                if first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              input_shape=(4*24000, 1), padding='same', weights=new_weights)
                    first_layer = False

                elif not first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              padding='same', weights=new_weights)

                new_model.add(new_layer)


            elif method == 'dilate':

                new_kernels = dilate_kernels(old_kernels)
                new_weights = [utils.get_weights(new_kernels), biases]

                if first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              input_shape=(4*24000, 1), padding='same', weights=new_weights)
                    first_layer = False

                elif not first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              padding='same', weights=new_weights)

                new_model.add(new_layer)

        elif type(layer) is keras.layers.pooling.MaxPooling1D:

            pool_size = layer.pool_size[0]
            new_model.add(layers.MaxPooling1D(pool_size=pool_size))

        elif type(layer) is keras.layers.pooling.AveragePooling1D:

            pool_size = layer.pool_size[0]
            new_model.add(layers.AveragePooling1D(pool_size=pool_size))

        elif type(layer) is keras.layers.Flatten:

            new_model.add(layers.Flatten())
            f_dim = layer.input_shape

        elif type(layer) is keras.layers.Dense:

            original_shape = layer.get_weights()[0].shape
            output_dim = layer.get_weights()[1].shape[0]
            shape = (f_dim[1], f_dim[2], output_dim)
            weights, biases = layer.get_weights()

            old_conv_weights = weights.reshape(shape)

            old_kernels = utils.get_kernels(old_conv_weights)

            if method == 'nearest_neighbor':

                new_kernels = nearest_neighbor(old_kernels)
                new_kernels = pad_zeros(new_kernels, old_kernels.shape[-1])
                new_conv_weights = utils.get_weights(new_kernels)
                new_dense_weights = [new_conv_weights.reshape((original_shape[0]*2,output_dim)), biases]

                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

            elif method == 'linear':
                new_kernels = linear(old_kernels)
                new_kernels = pad_zeros(new_kernels, old_kernels.shape[-1])
                new_conv_weights = utils.get_weights(new_kernels)
                new_dense_weights = [new_conv_weights.reshape(original_shape[0]*2,output_dim), biases]

                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

            elif method == 'distance_weighting':

                new_kernels = distance_weighting(old_kernels)
                new_kernels = pad_zeros(new_kernels, old_kernels.shape[-1])
                new_conv_weights = utils.get_weights(new_kernels)
                new_dense_weights = [new_conv_weights.reshape((original_shape[0]*2,output_dim)), biases]

                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

            elif method == 'dilate':

                new_kernels = dilate_kernels(old_kernels)
                new_kernels = pad_zeros(new_kernels, old_kernels.shape[-1])
                new_conv_weights = utils.get_weights(new_kernels)
                new_dense_weights = [new_conv_weights.reshape((original_shape[0]*2,output_dim)), biases]

                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

    X, Y = utils.load_data('Dataframes/Testing24.pickle')
    score = utils.test_model(new_model, X, Y)

    # SAVE MODEL
    # utils.save_model(new_model, 'Multiscaled/{}'.format(new_model_name))

    return new_model


def nearest_neighbor(old_kernels):

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    coefficient_matrix = []

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):

            old_kernel = old_kernels[current_node][prev_node]

            if old_kernel_size % 2 == 1:

                if ((old_kernel_size + 1) / 2) % 2 == 0:

                    if len(coefficient_matrix) == 0:

                        equations = []

                        new_kernel_size = old_kernel_size*2 - 3

                        new_kernel = sp.symbols('x0:%d' % new_kernel_size)

                        equation = sp.Eq(old_kernel[0], 0.5 * new_kernel[0])
                        equations.append(equation)

                        j = 0
                        for i in range(1, new_kernel_size - 1):
                            equation = sp.Eq(old_kernel[i], 0.5 * new_kernel[j] + new_kernel[j + 1] + 0.5 * new_kernel[j + 2])
                            equations.append(equation)
                            j += 2

                        equation = sp.Eq(old_kernel[-1], 0.5 * new_kernel[-1])
                        equations.append(equation)

                        coefficient_matrix = np.array(sp.linear_eq_to_matrix(equations, new_kernel)[0], dtype='float')
                        del equations

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

                elif ((old_kernel_size + 1) / 2) % 2 == 1:

                    if len(coefficient_matrix) == 0:

                        equations = []

                        new_kernel_size = old_kernel_size*2-1

                        new_kernel = sp.symbols('x0:%d' % new_kernel_size)

                        equation = sp.Eq(old_kernel[0], new_kernel[0] + 0.5 * new_kernel[1])
                        equations.append(equation)

                        j = 1
                        for i in range(1, old_kernel_size - 1):
                            equation = sp.Eq(old_kernel[i], 0.5 * new_kernel[j] + new_kernel[j + 1] + 0.5 * new_kernel[j + 2])
                            equations.append(equation)
                            j += 2

                        equation = sp.Eq(old_kernel[-1], new_kernel[-1] + 0.5 * new_kernel[-2])
                        equations.append(equation)

                        coefficient_matrix = np.array(sp.linear_eq_to_matrix(equations, new_kernel)[0], dtype='float')
                        del equations

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]

                    current_node_new_kernels.append(new_kernel)

            elif old_kernel_size % 2 == 0:

                if (old_kernel_size / 2) % 2 == 0:

                    if type(coefficient_matrix) is not scipy.sparse.coo.coo_matrix:

                        new_kernel_size = 2 * old_kernel_size - 1

                        # row indices
                        row_ind = []
                        # column indices
                        col_ind = []
                        # data to be stored in COO sparse matrix
                        data = []

                        # coefficient_matrix[0][0] = 1
                        row_ind.append(0)
                        col_ind.append(0)
                        data.append(1)

                        # coefficient_matrix[0][1] = 0.5
                        row_ind.append(0)
                        col_ind.append(1)
                        data.append(0.5)

                        j = 1
                        for i in range(1, old_kernel_size - 1):
                            # coefficient_matrix[i][j] = 0.5
                            row_ind.append(i)
                            col_ind.append(j)
                            data.append(0.5)

                            # coefficient_matrix[i][j + 1] = 1
                            row_ind.append(i)
                            col_ind.append(j+1)
                            data.append(1)

                            # coefficient_matrix[i][j + 2] = 0.5
                            row_ind.append(i)
                            col_ind.append(j + 2)
                            data.append(0.5)

                            j += 2

                        # coefficient_matrix[-1][-1] = 0.5
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 1)
                        data.append(0.5)

                        data = np.array(data, dtype='float')
                        row_ind = np.array(row_ind)
                        col_ind = np.array(col_ind)

                        coefficient_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))


                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]

                    current_node_new_kernels.append(new_kernel)

                elif (old_kernel_size / 2) % 2 == 1:

                    if type(coefficient_matrix) is not scipy.sparse.coo.coo_matrix:
                        
                        new_kernel_size = 2 * old_kernel_size -1

                        # row indices
                        row_ind = []
                        # column indices
                        col_ind = []
                        # data to be stored in COO sparse matrix
                        data = []

                        # coefficient_matrix[0][0] = 0.5
                        row_ind.append(0)
                        col_ind.append(0)
                        data.append(0.5)

                        j = 0
                        for i in range(1, old_kernel_size - 1):
                            # coefficient_matrix[i][j] = 0.5
                            row_ind.append(i)
                            col_ind.append(j)
                            data.append(0.5)

                            # coefficient_matrix[i][j + 1] = 1
                            row_ind.append(i)
                            col_ind.append(j+1)
                            data.append(1)

                            # coefficient_matrix[i][j + 2] = 0.5
                            row_ind.append(i)
                            col_ind.append(j + 2)
                            data.append(0.5)

                            j += 2

                        # coefficient_matrix[-1][-1] = 1
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 1)
                        data.append(1)

                        # coefficient_matrix[-1][-2] = 1
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 2)
                        data.append(0.5)

                        data = np.array(data, dtype='float')
                        row_ind = np.array(row_ind)
                        col_ind = np.array(col_ind)

                        coefficient_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)


def linear(old_kernels):

    coefficient_matrix = []

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):
            old_kernel = old_kernels[current_node][prev_node]

            if old_kernel_size % 2 == 1:

                if (old_kernel_size + 1) / 2 % 2 == 0:

                    if len(coefficient_matrix) == 0:

                        equations = []

                        new_kernel_size = old_kernel_size*2 - 3

                        new_kernel = sp.symbols('x0:%d' % new_kernel_size)

                        equation = sp.Eq(old_kernel[0], new_kernel[1] + 1.5 * new_kernel[0])
                        equations.append(equation)

                        j = 0
                        for i in range(1, old_kernel_size - 2):
                            equation = sp.Eq(old_kernel[i], new_kernel[j+3] + 1.5*new_kernel[j+2] - 0.5 * new_kernel[j])
                            equations.append(equation)
                            j += 2

                        equation = sp.Eq(old_kernel[-2], 1.5*new_kernel[-1] - 0.5 * new_kernel[-3])
                        equations.append(equation)

                        equation = sp.Eq(old_kernel[-1], -0.5 * new_kernel[-1])
                        equations.append(equation)

                        coefficient_matrix = np.array(sp.linear_eq_to_matrix(equations, new_kernel)[0], dtype='float')
                        del equations

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

                elif (old_kernel_size + 1) / 2 % 2 == 1:

                    if len(coefficient_matrix) == 0:

                        equations = []

                        new_kernel_size = old_kernel_size*2-2

                        new_kernel = sp.symbols('x0:%d' % new_kernel_size)

                        equation = sp.Eq(old_kernel[0], new_kernel[0])
                        equations.append(equation)
                        equation = sp.Eq(old_kernel[1], 0.5 * new_kernel[1] + new_kernel[2])
                        equations.append(equation)

                        j = 1
                        for i in range(2, old_kernel_size - 1):
                            equation = sp.Eq(old_kernel[i], new_kernel[j+3] + 1.5*new_kernel[j+2] - 0.5 * new_kernel[j])
                            equations.append(equation)
                            j += 2

                        equation = sp.Eq(old_kernel[-1], - 0.5 * new_kernel[-2])
                        equations.append(equation)

                        coefficient_matrix = np.array(sp.linear_eq_to_matrix(equations, new_kernel)[0], dtype='float')
                        del equations

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

            elif old_kernel_size % 2 == 0:

                if (old_kernel_size / 2) % 2 == 0:

                    if type(coefficient_matrix) is not scipy.sparse.coo.coo_matrix:

                        new_kernel_size = 2 * old_kernel_size - 2

                        # row indices
                        row_ind = []
                        # column indices
                        col_ind = []
                        # data to be stored in COO sparse matrix
                        data = []

                        # coefficient_matrix[0][0] = 1
                        row_ind.append(0)
                        col_ind.append(0)
                        data.append(1)

                        # coefficient_matrix[1][1] = 1.5
                        row_ind.append(1)
                        col_ind.append(1)
                        data.append(1.5)

                        # coefficient_matrix[1][2] = 1
                        row_ind.append(1)
                        col_ind.append(2)
                        data.append(1)

                        j = 1
                        for i in range(2, old_kernel_size - 2):
                            # coefficient_matrix[i][j] = -0.5
                            row_ind.append(i)
                            col_ind.append(j)
                            data.append(-0.5)

                            # coefficient_matrix[i][j + 2] = 1.5
                            row_ind.append(i)
                            col_ind.append(j + 2)
                            data.append(1.5)

                            # coefficient_matrix[i][j + 3] = 1
                            row_ind.append(i)
                            col_ind.append(j + 3)
                            data.append(1)

                            j += 2

                        # coefficient_matrix[-1][-1] = -0.5
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 1)
                        data.append(-0.5)

                        # coefficient_matrix[-2][-3] = 1.5
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 1)
                        data.append(1.5)

                        # coefficient_matrix[-2][-1] = -0.5
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 3)
                        data.append(-0.5)

                        data = np.array(data, dtype='float')
                        row_ind = np.array(row_ind)
                        col_ind = np.array(col_ind)

                        coefficient_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]

                    current_node_new_kernels.append(new_kernel)

                elif (old_kernel_size / 2) % 2 == 1:

                    if type(coefficient_matrix) is not scipy.sparse.coo.coo_matrix:

                        new_kernel_size = 2 * old_kernel_size - 1

                        # row indices
                        row_ind = []
                        # column indices
                        col_ind = []
                        # data to be stored in COO sparse matrix
                        data = []

                        # coefficient_matrix[0][0] = 1.5
                        row_ind.append(0)
                        col_ind.append(0)
                        data.append(1.5)

                        # coefficient_matrix[0][1] = 1
                        row_ind.append(0)
                        col_ind.append(0)
                        data.append(1)

                        j = 0
                        for i in range(1, old_kernel_size - 1):
                            # coefficient_matrix[i][j] = -0.5
                            row_ind.append(i)
                            col_ind.append(j)
                            data.append(-0.5)

                            # coefficient_matrix[i][j + 2] = 1.5
                            row_ind.append(i)
                            col_ind.append(j + 2)
                            data.append(1.5)

                            # coefficient_matrix[i][j + 3] = 1
                            row_ind.append(i)
                            col_ind.append(j + 3)
                            data.append(1)

                            j += 2

                        # coefficient_matrix[-1][-2] = -0.5
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 2)
                        data.append(-0.5)

                        data = np.array(data, dtype='float')
                        row_ind = np.array(row_ind)
                        col_ind = np.array(col_ind)

                        coefficient_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)


def distance_weighting(old_kernels):

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    coefficient_matrix = []

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):
            old_kernel = old_kernels[current_node][prev_node]

            if old_kernel_size % 2 == 1:

                if (old_kernel_size + 1) / 2 % 2 == 0:

                    if len(coefficient_matrix) == 0:

                        equations = []

                        new_kernel_size = old_kernel_size*2 - 5

                        new_kernel = sp.symbols('x0:%d' % new_kernel_size)

                        equation = sp.Eq(old_kernel[0], -1/8 * new_kernel[0])
                        equations.append(equation)

                        equation = sp.Eq(old_kernel[1], -1 / 8 * new_kernel[3] + 3/4 * new_kernel[1])
                        equations.append(equation)

                        j = 0
                        for i in range(2, old_kernel_size - 2):
                            equation = sp.Eq(old_kernel[i], -1 / 8 * new_kernel[j+4] + 3/4 * new_kernel[j+2] + new_kernel[j+1] + 3/8 * new_kernel[j])
                            equations.append(equation)
                            j += 2

                        equation = sp.Eq(old_kernel[-2], 3/4 * new_kernel[-1] + new_kernel[-2] + 3/8 * new_kernel[-3])
                        equations.append(equation)

                        equation = sp.Eq(old_kernel[-1], 3/8 * new_kernel[-1])
                        equations.append(equation)

                        coefficient_matrix = np.array(sp.linear_eq_to_matrix(equations, new_kernel)[0], dtype='float')
                        del equations

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

                elif (old_kernel_size + 1) / 2 % 2 == 1:

                    if len(coefficient_matrix) == 0:

                        equations = []

                        new_kernel_size = old_kernel_size*2 - 3

                        new_kernel = sp.symbols('x0:%d' % new_kernel_size)

                        equation = sp.Eq(old_kernel[0], -1/8 * new_kernel[1])
                        equations.append(equation)
                        equation = sp.Eq(old_kernel[1], new_kernel[0] + 3/4 * new_kernel[1] - 1/8 * new_kernel[3])
                        equations.append(equation)

                        j = 1
                        for i in range(2, old_kernel_size - 2):
                            equation = sp.Eq(old_kernel[i], - 1/8 * new_kernel[j+4] + 3/4 * new_kernel[j+2] + new_kernel[j+1] + 3/8 * new_kernel[j])
                            equations.append(equation)
                            j += 2

                        equation = sp.Eq(old_kernel[-2], 3/4 * new_kernel[-2] + new_kernel[-3] + 3/8 * new_kernel[-4])
                        equations.append(equation)

                        equation = sp.Eq(old_kernel[-1], new_kernel[-1] + 3/8 * new_kernel[-2])
                        equations.append(equation)

                        coefficient_matrix = np.array(sp.linear_eq_to_matrix(equations, new_kernel)[0], dtype='float')
                        del equations

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

            elif old_kernel_size % 2 == 0:

                if (old_kernel_size / 2) % 2 == 0:

                    if type(coefficient_matrix) is not scipy.sparse.coo.coo_matrix:

                        new_kernel_size = 2 * old_kernel_size - 2

                        # row indices
                        row_ind = []
                        # column indices
                        col_ind = []
                        # data to be stored in COO sparse matrix
                        data = []

                        # coefficient_matrix[0][1] = -1/8
                        row_ind.append(0)
                        col_ind.append(1)
                        data.append(-1/8)

                        # coefficient_matrix[1][0] = 1
                        row_ind.append(1)
                        col_ind.append(0)
                        data.append(1)

                        # coefficient_matrix[1][1] = 3/4
                        row_ind.append(1)
                        col_ind.append(1)
                        data.append(3/4)

                        # coefficient_matrix[1][3] = -1/8
                        row_ind.append(1)
                        col_ind.append(3)
                        data.append(-1/8)

                        j = 1
                        for i in range(2, old_kernel_size - 2):
                            # coefficient_matrix[i][j] = 3/8
                            row_ind.append(i)
                            col_ind.append(j)
                            data.append(3/8)

                            # coefficient_matrix[i][j + 1] = 1
                            row_ind.append(i)
                            col_ind.append(j + 1)
                            data.append(1)

                            # coefficient_matrix[i][j + 2] = 3/4
                            row_ind.append(i)
                            col_ind.append(j + 2)
                            data.append(3/4)

                            # coefficient_matrix[i][j + 4] = -1/8
                            row_ind.append(i)
                            col_ind.append(j + 4)
                            data.append(-1/8)

                            j += 2

                        # coefficient_matrix[-1][-1] = 3/8
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 1)
                        data.append(3/8)

                        # coefficient_matrix[-2][-1] = 3/4
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 1)
                        data.append(3/4)

                        # coefficient_matrix[-2][-2] = -1
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 2)
                        data.append(1)

                        # coefficient_matrix[-2][-3] = 3/8
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 3)
                        data.append(3/8)

                        data = np.array(data, dtype='float')
                        row_ind = np.array(row_ind)
                        col_ind = np.array(col_ind)

                        coefficient_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]

                    current_node_new_kernels.append(new_kernel)

                elif (old_kernel_size / 2) % 2 == 1:

                    if type(coefficient_matrix) is not scipy.sparse.coo.coo_matrix:

                        new_kernel_size = 2 * old_kernel_size - 2

                        # row indices
                        row_ind = []
                        # column indices
                        col_ind = []
                        # data to be stored in COO sparse matrix
                        data = []

                        # coefficient_matrix[0][0] = -1/8
                        row_ind.append(0)
                        col_ind.append(0)
                        data.append(-1/8)

                        # coefficient_matrix[1][0] = 3/4
                        row_ind.append(1)
                        col_ind.append(0)
                        data.append(3/4)

                        # coefficient_matrix[1][2] = -1/8
                        row_ind.append(1)
                        col_ind.append(2)
                        data.append(-1/8)

                        j = 0
                        for i in range(2, old_kernel_size - 2):
                            # coefficient_matrix[i][j] = 3/8
                            row_ind.append(i)
                            col_ind.append(j)
                            data.append(3/8)

                            # coefficient_matrix[i][j + 1] = 1
                            row_ind.append(i)
                            col_ind.append(j + 1)
                            data.append(1)

                            # coefficient_matrix[i][j + 2] = 3/4
                            row_ind.append(i)
                            col_ind.append(j + 2)
                            data.append(3 / 4)

                            # coefficient_matrix[i][j + 4] = -1/8
                            row_ind.append(i)
                            col_ind.append(j + 4)
                            data.append(-1 / 8)

                            j += 2

                        # coefficient_matrix[-1][-1] = 1
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 1)
                        data.append(1)

                        # coefficient_matrix[-1][-2] = 3/8
                        row_ind.append(old_kernel_size - 1)
                        col_ind.append(new_kernel_size - 2)
                        data.append(3/8)

                        # coefficient_matrix[-2][-2] = 3/4
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 2)
                        data.append(3 / 4)

                        # coefficient_matrix[-2][-3] = -1
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 3)
                        data.append(1)

                        # coefficient_matrix[-2][-4] = 3/8
                        row_ind.append(old_kernel_size - 2)
                        col_ind.append(new_kernel_size - 4)
                        data.append(3 / 8)

                        data = np.array(data, dtype='float')
                        row_ind = np.array(row_ind)
                        col_ind = np.array(col_ind)

                        coefficient_matrix = sparse.coo_matrix((data, (row_ind, col_ind)))

                    new_kernel = lsqr(coefficient_matrix, old_kernel)[0]
                    current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)


def dilate_kernels(old_kernels, rate=2):

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):

            old_kernel = old_kernels[current_node][prev_node]

            new_kernel_size = old_kernel_size * 2 - 1

            new_kernel = np.zeros(shape=(new_kernel_size))

            new_kernel.put([i for i in range(0,new_kernel_size,rate)], old_kernel)

            current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)



def pad_zeros(new_kernels, old_kernel_size):

        new_kernel_size = new_kernels.shape[-1]

        zeros = old_kernel_size*2 - new_kernel_size

        if zeros > 0:

            new_kernels_padded = np.ndarray(shape=(new_kernels.shape[0], new_kernels.shape[1], new_kernels.shape[2] + zeros))

            for i in range(new_kernels.shape[0]):
                for j in range(new_kernels.shape[1]):
                    new_kernels_padded[i][j] = np.append(new_kernels[i][j], np.zeros(zeros))

            return new_kernels_padded

        elif zeros < 0:

            new_kernels_padded = np.delete(new_kernels, slice(zeros,None), axis=2)
            return new_kernels_padded

        else:
            return new_kernels


if __name__ == '__main__':

    upscale('dilate', 'Model_12KHz_98%_meanPooling', 'test')
