from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM

def create_neural_network(freq_dimensions, hidden_dimensions, rec_units=1):
  model = Sequential()
  model.add(TimeDistributedDense(input_dim=freq_dimensions, output_dim=hidden_dimensions))
  for i in range(rec_units) :
    model.add(LSTM(input_dim= freq_dimensions, output_dim=hidden_dimensions, return_sequences=True))

  model.add(TimeDistributedDense(input_dim=hidden_dimensions, output_dim=freq_dimensions))  
  model.compile(loss='mean_squared_error', optimizer='rmsprop')
  return model


