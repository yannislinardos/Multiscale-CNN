from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dropout, Conv1D, MaxPooling1D, Flatten, Dense, AveragePooling1D
from keras.callbacks import TensorBoard
import numpy as np
import pickle
import gc
import os

gc.enable()

print('TRAINING FOR 12 KHZ')

f = open("Failed.txt", "a")
f.write('-------------------------------------FAILURES AT 12 KHZ----------------------------------------\n')
f.close()

input_dim = 4*12*1000

# Load Data
pickle_in = open('Dataframes/Training12.pickle', 'rb')
data = pickle.load(pickle_in)
pickle_in.close()

X = np.array(data['data'])
Y = np.array(data['class'])

# Reshape
X = np.stack(X)
X = np.expand_dims(X, axis=3)
Y = to_categorical(Y, 5)

# Memory
del data
gc.collect()

# What we will test
dense_layers = [0, 1]
layer_sizes = [32, 64]
conv_layers = [1, 2, 3, 4, 5]
kernel_sizes = [3, 5]
dropout_rates = [0.3, 0.5]
pooling = ['max', 'mean']
pool_sizes = [2, 4]


# Heuristics
# for pool_size in pool_sizes:
#     for pool in pooling:
#         for kernel_size in kernel_sizes:
#             for dropout_rate in dropout_rates:
#                 for dense_layer in dense_layers:
#                     for layer_size in layer_sizes:
#                         for conv_layer in conv_layers:
#
#                             try:
#                                 NAME = '{} conv, {} nodes, {} dense, {} kernel_size, {} dropout, {} pooling, {} pool_size'.format(
#                                     conv_layer, layer_size, dense_layer, kernel_size, dropout_rate, pool, pool_size)
#                                 print(NAME)
#
#                                 if os.path.exists(os.path.join('logs12KHz',NAME)):
#                                     break
#
#                                 tensorboard = TensorBoard(log_dir='logs12KHz/{}'.format(NAME))
#
#                                 model = Sequential()
#                                 model.add(Conv1D(layer_size, kernel_size=kernel_size, activation='relu', input_shape=(input_dim, 1)))
#
#                                 for l in range(conv_layer-1):
#                                     model.add(Conv1D(layer_size, kernel_size=kernel_size, activation='relu'))
#
#                                     if pool=='max':
#                                         model.add(MaxPooling1D(pool_size=pool_size))
#                                     elif pool=='mean':
#                                         model.add(AveragePooling1D(pool_size=pool_size))
#
#
#                                 model.add(Dropout(dropout_rate))
#
#                                 model.add(Flatten())
#
#                                 for l in range(dense_layer):
#                                     model.add(Dense(layer_size, activation='relu'))
#
#                                 model.add(Dense(5, activation='sigmoid'))
#                                 #model.summary()
#
#                                 model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#                                 # validation 0.15*0.85
#                                 model.fit(X, Y, batch_size=5, epochs=10, shuffle=True, validation_split=0.12, callbacks=[tensorboard])
#
#                             except Exception:
#                                 f = open("Failed.txt", "a")
#                                 f.write('Failed at {}\n'.format(NAME))
#                                 f.close()
#                                 pass

#####################################################################################

print('TRAINING FOR 48 KHZ')

input_dim = 4 * 48 * 1000

f = open("Failed.txt", "a")
f.write('-------------------------------------FAILURES AT 48 KHZ----------------------------------------\n')
f.close()

# Load Data
pickle_in = open('Dataframes/Training48.pickle', 'rb')
data = pickle.load(pickle_in)
pickle_in.close()

X = np.array(data['data'])
Y = np.array(data['class'])

# Reshape
X = np.stack(X)
X = np.expand_dims(X, axis=3)
Y = to_categorical(Y, 5)

# Memory
del data
gc.collect()

# Heuristics
for pool_size in pool_sizes:
    for pool in pooling:
        for kernel_size in kernel_sizes:
            for dropout_rate in dropout_rates:
                for dense_layer in dense_layers:
                    for layer_size in layer_sizes:
                        for conv_layer in conv_layers:
                            try:
                                NAME = '{} conv, {} nodes, {} dense, {} kernel_size, {} dropout, {} pooling, {} pool_size'.format(
                                    conv_layer, layer_size, dense_layer, kernel_size, dropout_rate, pool, pool_size)
                                print(NAME)

                                tensorboard = TensorBoard(log_dir='logs48KHz/{}'.format(NAME))

                                model = Sequential()
                                model.add(Conv1D(layer_size, kernel_size=kernel_size, activation='relu',
                                                 input_shape=(input_dim, 1)))

                                for l in range(conv_layer - 1):
                                    model.add(Conv1D(layer_size, kernel_size=kernel_size, activation='relu'))

                                    if pool == 'max':
                                        model.add(MaxPooling1D(pool_size=pool_size))
                                    elif pool == 'mean':
                                        model.add(AveragePooling1D(pool_size=pool_size))

                                model.add(Dropout(dropout_rate))

                                model.add(Flatten())

                                for l in range(dense_layer):
                                    model.add(Dense(layer_size, activation='relu'))

                                model.add(Dense(5, activation='sigmoid'))
                                # model.summary()

                                model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

                                # validation 0.15*0.85
                                model.fit(X, Y, batch_size=5, epochs=10, shuffle=True, validation_split=0.12,
                                          callbacks=[tensorboard])

                            except Exception:
                                f = open("Failed.txt", "a")
                                f.write('Failed at {}\n'.format(NAME))
                                f.close()
                                pass

