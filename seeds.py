import numpy as np

#A very simple seed generator
#Copies a random example's first seed_length sequences as input to the generation algorithm
def generate_copy_seed_sequence(seed_length, training_data):
  num_examples = training_data.shape[0]
  example_len = training_data.shape[1]
  randIdx = np.random.randint(num_examples, size=1)[0]
  randSeed = np.concatenate(tuple([training_data[randIdx + i] for i in range(seed_length)]), axis=0)
  seedSeq = np.reshape(randSeed, (1, randSeed.shape[0], randSeed.shape[1]))
  return seedSeq

def generate_from_seed(model, seed, sequence_length, data_variance, data_mean):
  seedSeq = seed.copy()
  output = []
  for it in range(sequence_length):
    seedSeqNew = model._predict(seedSeq) 
    if it == 0:
      for i in range(seedSeqNew.shape[1]):
        output.append(seedSeqNew[0][i].copy())
    else:
      output.append(seedSeqNew[0][seedSeqNew.shape[1]-1].copy()) 
    newSeq = seedSeqNew[0][seedSeqNew.shape[1]-1]
    newSeq = np.reshape(newSeq, (1, 1, newSeq.shape[0]))
    seedSeq = np.concatenate((seedSeq, newSeq), axis=1)
  for i in range(len(output)):
	  output[i] *= data_variance
	  output[i] += data_mean
  return output