import utils
import keras
from keras import models
from keras import layers
import numpy as np



def downscale(method: str, old_model_name: str, new_model_name: str, avg_pool=False):

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
                                          input_shape=(48000, 1), padding='same', weights=new_weights)
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
                                              input_shape=(48000, 1), padding='same', weights=new_weights)
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
                                              input_shape=(48000, 1), padding='same', weights=new_weights)
                    first_layer = False

                elif not first_layer:
                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              padding='same', weights=new_weights)

                new_model.add(new_layer)

        elif type(layer) is keras.layers.pooling.MaxPooling1D:

            pool_size = layer.pool_size[0]

            new_model.add(layers.MaxPooling1D(pool_size=pool_size))

        elif type(layer) is keras.layers.pooling.AveragePooling1D:

            nodes = layer.get_output_at(0).shape[-1].value
            pool_size = layer.pool_size[0]

            if method == 'nearest_neighbor':

                new_model.add(layers.AveragePooling1D(pool_size=pool_size))

            elif method == 'linear':

                if avg_pool is True:
                    new_model.add(layers.AveragePooling1D(pool_size=pool_size))

                else:
                    new_kernels = down_scale_avg_pooling(nodes, [3/2,-1/4,-1/4])
                    dummy_bias = np.zeros(nodes)
                    new_weights = [utils.get_weights(new_kernels), dummy_bias]

                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[-1], activation='linear'
                                              , padding='same', strides=2, weights=new_weights)

                    new_model.add(new_layer)

            elif method == 'distance_weighting':

                if avg_pool is True:
                    new_model.add(layers.AveragePooling1D(pool_size=pool_size))

                else:
                    new_kernels = down_scale_avg_pooling(nodes, [-1/4,1/2,3/4])
                    dummy_bias = np.zeros(nodes)
                    new_weights = [utils.get_weights(new_kernels), dummy_bias]

                    new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[-1], activation='linear',
                                              padding='same', strides=2, weights=new_weights)
                    new_model.add(new_layer)

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
                new_dense_weights = [new_conv_weights.reshape((original_shape[0]//2,output_dim)), biases]

                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

            elif method == 'linear':
                new_kernels = linear(old_kernels)
                new_kernels = pad_zeros(new_kernels, old_kernels.shape[-1])
                new_conv_weights = utils.get_weights(new_kernels)
                new_dense_weights = [new_conv_weights.reshape(original_shape[0]//2,output_dim), biases]
                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

            elif method == 'distance_weighting':

                new_kernels = distance_weighting(old_kernels)
                new_kernels = pad_zeros(new_kernels, old_kernels.shape[-1])
                new_conv_weights = utils.get_weights(new_kernels)
                new_dense_weights = [new_conv_weights.reshape((original_shape[0]//2,output_dim)), biases]

                new_model.add(layers.Dense(output_dim, activation=layer.activation, weights=new_dense_weights))

    X, Y = utils.load_data('Dataframes/Testing12.pickle')
    score = utils.test_model(new_model, X, Y)

    # SAVE MODEL
    # utils.save_model(new_model, 'Multiscaled/{}'.format(new_model_name))


def down_scale_avg_pooling(nodes, kernel):

    arr = np.zeros(shape=(nodes, nodes, len(kernel)))

    arr[:][:] = np.array(kernel)

    return arr


def nearest_neighbor(old_kernels):

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):

            old_kernel = old_kernels[current_node][prev_node]

            if old_kernel_size % 2 == 1:

                if ((old_kernel_size + 1) / 2) % 2 == 0:

                    new_kernel_size = (old_kernel_size+3)//2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = 0.5 * old_kernel[0]

                    j = 0
                    for i in range(1, new_kernel_size - 1):
                        new_kernel[i] = 0.5 * old_kernel[j] + old_kernel[j + 1] + 0.5 * old_kernel[j + 2]
                        j += 2

                    new_kernel[-1] = 0.5 * old_kernel[-1]

                    current_node_new_kernels.append(new_kernel)

                elif ((old_kernel_size + 1) / 2) % 2 == 1:

                    new_kernel_size = (old_kernel_size+1)//2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = old_kernel[0] + 0.5 * old_kernel[1]

                    j = 1
                    for i in range(1, new_kernel_size - 1):
                        new_kernel[i] = 0.5 * old_kernel[j] + old_kernel[j + 1] + 0.5 * old_kernel[j + 2]
                        j += 2

                    new_kernel[-1] = old_kernel[-1] + 0.5 * old_kernel[-2]

                    current_node_new_kernels.append(new_kernel)

            elif old_kernel_size % 2 == 0:

                if ((old_kernel_size) / 2) % 2 == 1:

                    new_kernel_size = old_kernel_size//2 + 1

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = 0.5 * old_kernel[0]

                    j = 0
                    for i in range(1, new_kernel_size - 1):
                        new_kernel[i] = 0.5 * old_kernel[j] + old_kernel[j + 1] + 0.5 * old_kernel[j + 2]
                        j += 2

                    new_kernel[-1] = 0.5 * old_kernel[-1]

                    current_node_new_kernels.append(new_kernel)

                elif (old_kernel_size / 2) % 2 == 0:

                    new_kernel_size = old_kernel_size//2 + 1

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = old_kernel[0] + 0.5 * old_kernel[1]

                    j = 1
                    for i in range(1, new_kernel_size - 1):
                        new_kernel[i] = 0.5 * old_kernel[j] + old_kernel[j + 1] + 0.5 * old_kernel[j + 2]
                        j += 2

                    new_kernel[-1] = old_kernel[-1] + 0.5 * old_kernel[-2]

                    current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)


def linear(old_kernels):

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):
            old_kernel = old_kernels[current_node][prev_node]

            if old_kernel_size % 2 == 1:

                if (old_kernel_size + 1) / 2 % 2 == 0:

                    new_kernel_size = (old_kernel_size + 3) // 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = old_kernel[1] + 1.5 * old_kernel[0]

                    j = 0
                    for i in range(1, new_kernel_size - 2):
                        new_kernel[i] = old_kernel[j+3] + 1.5*old_kernel[j+2] - 0.5 * old_kernel[j]
                        j += 2

                    new_kernel[-2] = 1.5 * old_kernel[-1] - 0.5 * old_kernel[-3]
                    new_kernel[-1] = -0.5 * old_kernel[-1]

                elif (old_kernel_size + 1) / 2 % 2 == 1:

                    new_kernel_size = (old_kernel_size + 3) // 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = old_kernel[0]
                    new_kernel[1] = 0.5 * old_kernel[1] + old_kernel[2]

                    j = 1
                    for i in range(2, new_kernel_size - 2):
                        new_kernel[i] = old_kernel[j+3] + 1.5*old_kernel[j+2] - 0.5 * old_kernel[j]
                        j += 2

                    new_kernel[-1] = - 0.5 * old_kernel[-2]

            elif old_kernel_size % 2 == 0:

                if (old_kernel_size / 2) % 2 == 0:

                    new_kernel_size = old_kernel_size // 2 + 1

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = old_kernel[0]
                    new_kernel[1] = 1.5 * old_kernel[1] + old_kernel[2]

                    j = 1
                    for i in range(2, new_kernel_size - 1):
                        new_kernel[i] = old_kernel[j + 3] + 1.5 * old_kernel[j + 2] - 0.5 * old_kernel[j]
                        j += 2

                    new_kernel[-1] = - 0.5 * old_kernel[-1]

                elif (old_kernel_size / 2) % 2 == 1:

                    new_kernel_size = old_kernel_size//2 + 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = old_kernel[1] + 1.5 * old_kernel[0]

                    j = 0
                    for i in range(1, new_kernel_size - 1):
                        new_kernel[i] = old_kernel[j+3] + 1.5*old_kernel[j+2] - 0.5 * old_kernel[j]
                        j += 2

                    new_kernel[-1] = -0.5 * old_kernel[-2]

            current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)


def distance_weighting(old_kernels):

    all_new_kernels = []

    old_kernel_size = old_kernels.shape[2]

    for current_node in range(old_kernels.shape[0]):

        current_node_new_kernels = []

        for prev_node in range(old_kernels.shape[1]):
            old_kernel = old_kernels[current_node][prev_node]

            if old_kernel_size % 2 == 1:

                if (old_kernel_size + 1) / 2 % 2 == 0:

                    new_kernel_size = (old_kernel_size + 5) // 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = -1/8*old_kernel[0]
                    new_kernel[1] = -1/8* old_kernel[2] + 3/4 * old_kernel[0]

                    j = 0
                    for i in range(2, new_kernel_size - 2):
                        new_kernel[i] = -1/8*old_kernel[j+4] + 3/4 * old_kernel[j+2] + old_kernel[j -+1] + 3/8 * old_kernel[j]
                        j += 2

                    new_kernel[-2] = 3/4 * old_kernel[-1] + old_kernel[-2] + 3/8 * old_kernel[-3]
                    new_kernel[-1] = 3/8 * old_kernel[-1]

                elif (old_kernel_size + 1) / 2 % 2 == 1:

                    new_kernel_size = (old_kernel_size + 3) // 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = -1/8*old_kernel[1]
                    new_kernel[1] = - 1/8 * old_kernel[3] + 3/4 * old_kernel[1] + old_kernel[0]

                    j = 1
                    for i in range(2, new_kernel_size - 2):
                        new_kernel[i] = -1/8*old_kernel[j+4] + 3/4 * old_kernel[j+2] + old_kernel[j+1] + 3/8 * old_kernel[j]
                        j += 2

                    new_kernel[-2] = 3/4 * old_kernel[-2] + old_kernel[-3] + 3/8 * old_kernel[-4]
                    new_kernel[-1] = old_kernel[-1] + 3/8 * old_kernel[-2]

            elif old_kernel_size % 2 == 0:

                if (old_kernel_size / 2) % 2 == 1:

                    new_kernel_size = old_kernel_size//2 + 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = -1/8 * old_kernel[0]
                    new_kernel[1] = -1/8 * old_kernel[2] + 3 / 4 * old_kernel[0]

                    j = 0
                    for i in range(2, new_kernel_size - 2):
                        new_kernel[i] = -1/8 * old_kernel[j+4] + 3/4 * old_kernel[j+2] + old_kernel[j+1] + 3/8 * old_kernel[j]
                        j += 2

                    new_kernel[-2] = 3/4 * old_kernel[-2] + old_kernel[-3] + 3/8 * old_kernel[-4]
                    new_kernel[-1] = 3/8 * old_kernel[-2] + old_kernel[-1]

                elif (old_kernel_size / 2) % 2 == 0:

                    new_kernel_size = old_kernel_size//2 + 2

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = -1/8 * old_kernel[1]
                    new_kernel[1] = -1/8 * old_kernel[3] + 3/4 * old_kernel[1] + old_kernel[0]

                    j = 1
                    for i in range(2, new_kernel_size - 2):
                        new_kernel[i] = -1/8 * old_kernel[j+4] + 3/4 * old_kernel[j+2] + old_kernel[j+1] + 3/8 * old_kernel[j]
                        j += 2

                    new_kernel[-2] = 3/4 * old_kernel[-1] + old_kernel[-2] + 3/8 * old_kernel[-3]
                    new_kernel[-1] = 3/8 * old_kernel[-1]

            current_node_new_kernels.append(new_kernel)

        all_new_kernels.append(current_node_new_kernels)

    return np.array(all_new_kernels)


def pad_zeros(new_kernels, old_kernel_size):

        new_kernel_size = new_kernels.shape[-1]

        zeros = old_kernel_size//2 - new_kernel_size

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

    downscale('linear', 'Model_24KHz_97%_meanPooling', 'test', False)
