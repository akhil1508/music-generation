from __future__ import absolute_import
from __future__ import print_function
import numpy as np
import os
from train import create_neural_network
from seeds import generate_copy_seed_sequence, generate_from_seed
from conversions import *
import config as nn_config

config = nn_config.get_config()
sample_frequency = config['sampling_frequency']
inputFile = config['model_file']
model_basename = config['model_basename']
cur_iter = 25
model_filename = model_basename + str(cur_iter)
output_filename = './generated_song.wav'

print ('Loading training data')
X_train = np.load(inputFile + '_x.npy')
y_train = np.load(inputFile + '_y.npy')
X_mean = np.load(inputFile + '_mean.npy')
X_var = np.load(inputFile + '_var.npy')
print ('Finished loading training data')

freq_space_dims = X_train.shape[2]
hidden_dims = config['hidden_dimension_size']

model = create_neural_network(freq_dimensions=freq_space_dims, hidden_dimensions=hidden_dims)

if os.path.isfile(model_filename):
	model.load_weights(model_filename)
else:
	print('Model filename ' + model_filename + ' could not be found!')

print ('Starting generation!')
seed_len = 1
seed_seq = generate_copy_seed_sequence(seed_length=seed_len, training_data=X_train)

max_seq_len = 10; #Defines how long the final song is. Total song length in samples = max_seq_len * example_len
output = generate_from_seed(model=model, seed=seed_seq, 
	sequence_length=max_seq_len, data_variance=X_var, data_mean=X_mean)
print ('Finished generation!')

#Save the generated sequence to a WAV file
save_generated_example(output_filename, output, sample_frequency=sample_frequency)