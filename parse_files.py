import os
import scipy.io.wavfile as wav
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
sampling_freq = 44100
hidden_dimension_size = 1024

data = wav.read("./datasets/music/the_middle.wav")

arr = data[1]/(2**15)
model = Sequential()
model.add((512, input_shape=(arr.shape[0]),
return_sequences=True
))
model.add(Dropout(0.3))
model.add(LSTM(512, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(512))
model.add(Dense(256))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

return model