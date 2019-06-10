import numpy as np
import random
import os
from scipy import io
import scipy.io.wavfile
import pickle
import pandas as pd

DATADIR = '24KHz_dataset/Training'
CATEGORIES = ['microwave', 'mixer', 'sewing_machine', 'truck', 'vacuum_cleaner']
DATASIZE = scipy.io.wavfile.read('24KHz_dataset/Testing/microwave/130.wav')[1].size
filename = 'Training24.pickle'


# create training data
def create_training_data():
    training_data = []
    for category in CATEGORIES:
        path = os.path.join(DATADIR, category)
        class_num = CATEGORIES.index(category)

        for audio in os.listdir(path):
            audio_array = scipy.io.wavfile.read(os.path.join(path, audio))[1]
            training_data.append([audio_array, class_num])

    #random.shuffle(training_data)
    return training_data


# Pandas dataframe
def get_dataframe(training_data):
    df = pd.DataFrame(training_data, columns=['data', 'class'])
    return df


# X, Y pickles
def createPickle(training_data):

    X = []
    Y = []
    for array, label in training_data:
        X.append(array)
        Y.append(label)

    X = np.array(X).reshape(-1, DATASIZE, 1, 1)

    pickle_out = open('X.pickle', 'wb')
    pickle.dump(X, pickle_out)
    pickle_out.close()

    pickle_out = open('Y.pickle', 'wb')
    pickle.dump(Y, pickle_out)
    pickle_out.close()


if __name__ == "__main__":

    training_data = create_training_data()
    df = get_dataframe(training_data)

    pickle_out = open(os.path.join('Dataframes', filename), 'wb')
    pickle.dump(df, pickle_out)
    pickle_out.close()