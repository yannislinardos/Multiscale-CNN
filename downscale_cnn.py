import multiscale
import utils
import keras
from keras import models
from keras import layers
import numpy as np


class downscale_cnn:

    def __init__(self, method: str, old_model: str, new_model: str):

        self.method = method
        self.old_model_name = old_model
        self.new_model_name = new_model

        self.old_model = utils.load_model('Models/{}.yaml'.format(old_model), 'Models/{}.h5'.format(old_model))




    def downscale(self):

        new_model = models.Sequential()

        i = 0
        first_layer = True
        for layer in self.old_model.layers:
            print('we are in layer {}'.format(i))
            i += 1
            if type(layer) is keras.layers.convolutional.Conv1D:

                biases = layer.get_weights()[1]
                old_kernels = utils.get_kernels(layer.get_weights()[0])

                nodes = layer.kernel.shape[2].value

                if self.method == 'nearest_neighbor':

                    new_kernels = self.nearest_neighbor(self, old_kernels)
                    new_weights = [utils.get_weights(new_kernels), biases]

                    if first_layer:
                        new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              input_shape=(48000, 1), padding='same', weights=new_weights)
                        first_layer = False

                    elif not first_layer:
                        new_layer = layers.Conv1D(nodes, kernel_size=new_kernels.shape[2], activation=layer.activation,
                                              padding='same', weights=new_weights)


                    new_model.add(new_layer)

                elif self.method == 'linear':

                    new_kernels = self.linear(old_kernels)
                    new_weights = [utils.get_weights(new_kernels), biases]

                    new_layer = layers.Conv1D(nodes, kernel_size=len(new_kernels[0]), activation='relu',
                                              input_shape=(48000, 1), padding='same', weights=new_weights)
                    new_layer.set_weights(new_weights)

                    new_model.add(new_layer)

                elif self.method == 'distance_weighting':

                    new_kernels = self.distance_weighting(old_kernels)
                    new_weights = [utils.get_weights(new_kernels), biases]

                    new_layer = layers.Conv1D(nodes, kernel_size=len(new_kernels[0]), activation='relu',
                                              input_shape=(48000, 1), padding='same')
                    new_layer.set_weights(new_weights)

                    new_model.add(new_layer)


            elif type(layer) is keras.layers.pooling.AveragePooling1D:

                nodes = layer.output.shape[-1].value
                pool_size = layer.pool_size[0]

                if self.method == 'nearest_neighbor':

                    new_model.add(layers.AveragePooling1D(pool_size=pool_size))

                elif self.method == 'linear':

                    new_kernels = [[3/4,-1/4,-1/4]]
                    dummy_bias = np.array([0,0,0])
                    new_weights = [utils.get_weights(new_kernels), dummy_bias]

                    new_layer = layers.Conv1D(nodes, kernel_size=len(new_kernels[0]), activation='relu',
                                              input_shape=(48000, 1), padding='same', strides=2)
                    new_layer.set_weights(new_weights)

                    new_model.add(new_layer)

                elif self.method == 'distance_weighting':

                    new_kernels = [[-1/4,1/2,3/4]]
                    dummy_bias = np.array([0,0,0])
                    new_weights = [utils.get_weights(new_kernels), dummy_bias]

                    new_layer = layers.Conv1D(nodes, kernel_size=len(new_kernels[0]), activation='relu',
                                              input_shape=(48000, 1), padding='same', strides=2)
                    new_layer.set_weights(new_weights)

                    new_model.add(new_layer)


            elif type(layer) is keras.layers.pooling.Flatten:

                new_model.add(keras.layers.pooling.Flatten)


            elif type(layer) is keras.layers.core.Dense:



                new_model.add(layer)

        model_yaml = new_model.to_yaml()
        with open("Models/Multiscaled/{}.yaml".format(self.new_model_name), "w") as yaml_file:
            yaml_file.write(model_yaml)

        # serialize weights to HDF5
        new_model.save_weights("Models/Multiscaled/{}.h5".format(self.new_model_name))


    @staticmethod
    def nearest_neighbor(self, old_kernels):

        all_new_kernels = []

        old_kernel_size = old_kernels.shape[2]

        for current_node in range(old_kernels.shape[0]):

            current_node_new_kernels = []

            for prev_node in range(old_kernels.shape[1]):

                old_kernel = old_kernels[current_node][prev_node]


                if (old_kernel_size + 1) / 2 % 2 == 0:

                    new_kernel_size = (2 * (old_kernel_size + 1) / 4 - 1)

                    new_kernel = np.ndarray(shape=(new_kernel_size))

                    new_kernel[0] = 0.5 * old_kernel[0]

                    j = 0
                    for i in range(1, new_kernel_size - 1):
                        new_kernel[i] = 0.5 * old_kernel[j] + old_kernel[j + 1] + 0.5 * old_kernel[j + 2]
                        j += 2

                    new_kernel[-1] = 0.5 * old_kernel[-1]

                elif (old_kernel_size + 1) / 2 % 2 == 1:

                    new_kernel_size = (2 * (old_kernel_size + 3) // 4 - 1)

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


    @staticmethod
    def linear(old_kernels):

        new_kernels = []

        old_kernel_size = old_kernels.shape[1]

        for old_kernel in old_kernels:

            if (old_kernel_size + 1) / 2 % 2 == 0:

                new_kernel_size =  (old_kernel_size - 3) / 2

                new_kernel = np.ndarray(shape=(new_kernel_size))

                new_kernel[0] = -0.5 * old_kernel[-1]
                new_kernel[1] = 1.5 * old_kernel[-1]-0.5 * old_kernel[-3]

                j = 2
                for i in range(2, new_kernel_size - 2):
                    new_kernel[i] = old_kernel[-j] + 1.5*old_kernel[-j -1] - 0.5 * old_kernel[-j -3]
                    j += 2

                new_kernel[-1] = old_kernel[1] + 1.5 * old_kernel[2]

            elif (old_kernel_size + 1) / 2 % 2 == 1:

                new_kernel_size = (old_kernel_size - 1) // 2

                new_kernel = np.ndarray(shape=(new_kernel_size))

                new_kernel[0] = - 0.5 * old_kernel[-2]

                j = 1
                for i in range(1, new_kernel_size - 2):
                    new_kernel[i] = old_kernel[-j] + 1.5*old_kernel[-j -1] - 0.5 * old_kernel[-j -3]
                    j += 2

                new_kernel[-2] = 0.5 * old_kernel[1] + old_kernel[2]
                new_kernel[-1] = old_kernel[0]

            new_kernels.append(new_kernel)

        return new_kernels


    @staticmethod
    def distance_weighting(self, old_kernels):

        new_kernels = []

        old_kernel_size = old_kernels.shape[1]

        for old_kernel in old_kernels:

            if (old_kernel_size + 1) / 2 % 2 == 0:

                new_kernel_size = (old_kernel_size - 5) / 2

                new_kernel = np.ndarray(shape=(new_kernel_size))

                new_kernel[0] = 3/8 * old_kernel[-1]
                new_kernel[1] = 3/4 * old_kernel[-1] + old_kernel[-2] + 3/8 * old_kernel[-3]

                j = 1
                for i in range(2, new_kernel_size - 2):
                    new_kernel[i] = -1/8*old_kernel[-j] + 3/4 * old_kernel[-j - 3] + old_kernel[-j - 4] + 3/8 * old_kernel[-j-5]
                    j += 2

                new_kernel[-2] = -1/8* old_kernel[2] + 3/4 * old_kernel[0]
                new_kernel[-1] = -1/8*old_kernel[0]

            elif (old_kernel_size + 1) / 2 % 2 == 1:

                new_kernel_size = (old_kernel_size - 3) / 2

                new_kernel = np.ndarray(shape=(new_kernel_size))

                new_kernel[0] = -old_kernel[-1] + 3/8 * old_kernel[-2]
                new_kernel[1] = 3/4 * old_kernel[-2] + old_kernel[-3] + 3/8 * old_kernel[-4]


                j = 2
                for i in range(1, new_kernel_size - 2):
                    new_kernel[i] = -1/8*old_kernel[-j] + 3/4 * old_kernel[-j - 3] + old_kernel[-j - 4] + 3/8 * old_kernel[-j-5]
                    j += 2

                new_kernel[-2] = - 1/8 * old_kernel[3] + 3/4 * old_kernel[1] + old_kernel[0]
                new_kernel[-1] = -1/8*old_kernel[1]

            new_kernels.append(new_kernel)

        return new_kernels



if __name__ == '__main__':

    down = downscale_cnn('nearest_neighbor', 'Model_24KHz_87%_meanpooling', 'test')

    down.downscale()