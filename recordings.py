#import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
#import pandas as pd
#import sklearn
import librosa
from scipy import io
import scipy.io.wavfile
#import wave
import os
from pydub import AudioSegment
from shutil import copyfile
import random

# export data from wave
#bitrate, data = scipy.io.wavfile.read('Agrotiko.wav')


# Split the audios to segments
def split_wav(filename, segment_duration):

    newAudio = AudioSegment.from_wav(filename)
    duration = newAudio.duration_seconds
    number_of_segments = int(duration // segment_duration)

    print('It will be split in ' + str(number_of_segments) + " segments")

    dir = filename[0:-4]
    os.mkdir(dir)

    for i in range(0, number_of_segments):
        prev_second = segment_duration*i*1000
        next_second = segment_duration*(i+1)*1000
        segment = newAudio[prev_second:next_second]
        segment.export(dir+'/'+str(i)+'.wav', format="wav")


#downsample wave file
def downsample_audio(filename, new_rate):

    y, sr = librosa.load('OriginalRecordings/' + filename, sr=new_rate)
    librosa.output.write_wav('12KHz/' + str(new_rate)+'Hz_'+filename, y, sr)


#for filename in os.listdir('OriginalRecordings'):
#    downsample_audio(filename, 14000)

#for filename in os.listdir('14KHz'):
#    split_wav('14KHz/'+filename, 4)

# os.mkdir('OriginalRecordings/vacuum_cleaner')
# i = 0
# for filename in os.listdir('OriginalRecordings/vacuum_1m'):
#     copyfile('OriginalRecordings/vacuum_1m/'+filename, 'OriginalRecordings/vacuum_cleaner/'+str(i)+'.wav')
#     i = i + 1
#
# for filename in os.listdir('OriginalRecordings/vacuum_on_top'):
#     copyfile('OriginalRecordings/vacuum_on_top/'+filename, 'OriginalRecordings/vacuum_cleaner/'+str(i)+'.wav')
#     i = i + 1

# high_dataset = '48KHz_dataset'
# low_dataset = '12KHz_dataset'
# os.mkdir(low_dataset)
# new_rate = 12000
# for directory in os.listdir(high_dataset):
#     os.mkdir(low_dataset + '/' + directory)
#     for filename in os.listdir(high_dataset + '/' + directory):
#         y, sr = librosa.load(high_dataset + '/' + directory + '/' + filename, sr=new_rate)
#         librosa.output.write_wav(low_dataset + '/' + directory + '/' + filename, y, sr)


# shuffle data
# dataset = '48KHz_dataset'
# for directory in os.listdir(dataset):
#     max_num = len([name for name in os.listdir(os.path.join(dataset, directory))])
#     names_used = []
#     os.mkdir(os.path.join(dataset, directory, 'shuffled'))
#     for filename in os.listdir(os.path.join(dataset, directory)):
#         if filename != 'shuffled':
#             name = random.randrange(0, max_num)
#             while name in names_used:
#                 name = random.randrange(0, max_num)
#
#             copyfile(os.path.join(dataset, directory, filename), os.path.join(dataset, directory, 'shuffled', str(name) + '.wav'))
#             names_used.append(name)

high_dataset = '48KHz_dataset'
new_dataset = '24KHz_dataset'
new_rate = 24000
os.mkdir(new_dataset)
# loop over training and testing datasets
for set in os.listdir(high_dataset):
    os.mkdir(os.path.join(new_dataset, set))
    # loop over categories
    for directory in os.listdir(os.path.join(high_dataset, set)):
        os.mkdir(os.path.join(new_dataset, set, directory))
        for filename in os.listdir(os.path.join(high_dataset, set, directory)):
            y, sr = librosa.load(os.path.join(high_dataset, set, directory, filename), sr=new_rate)
            librosa.output.write_wav(os.path.join(new_dataset, set, directory, filename), y, sr)