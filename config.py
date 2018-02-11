def get_config() :
  params = {}
  params['sampling_frequency'] = 44100
  params['hidden_dimension_size'] = 1024
  params['model_basename'] = "./musicNPWeights"
  params['model_file'] = "./datasets/musicNP"
  params['dataset_directory'] = "./datasets/music/"
  return params