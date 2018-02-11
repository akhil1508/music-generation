from keras.models import Sequential
from keras.layers.core import TimeDistributedDense
from keras.layers.recurrent import LSTM
import numpy as np
import os
import config 

def create_neural_network(freq_dimensions, hidden_dimensions, rec_units=1):
  model = Sequential()
  model.add(TimeDistributedDense(input_dim=freq_dimensions, output_dim=hidden_dimensions))
  for i in range(rec_units) :
    model.add(LSTM(input_dim= freq_dimensions, output_dim=hidden_dimensions, return_sequences=True))

  model.add(TimeDistributedDense(input_dim=hidden_dimensions, output_dim=freq_dimensions))  
  model.compile(loss='mean_squared_error', optimizer='rmsprop')
  return model

configuration = config.get_config()
input_file = configuration['model_file'] 
cur_iter = 0
model_basename = configuration['model_basename']
model_filename = model_basename + str(cur_iter)
print ('Loading training data')
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
print ('Finished loading training data')

freq_space_dims = X_train.shape[2]
hidden_dims = config['hidden_dimension_size']

model = create_neural_network(freq_dimensions = freq_space_dims, hidden_dimensions=hidden_dims)

if os.path.isfile(model_filename):
  model.load_weights(model_filename)

num_iters = 50
num_epochs = 25
batch_size = 5

while (cur_iter < num_iters ) :
  print("Iteration: " + str(cur_iter))
  history = model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=epochs_per_iter, verbose=1, validation_split=0.0)
  cur_iter += num_epochs

model.save_weights(model_basename + str(cur_iter))