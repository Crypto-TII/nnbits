# !/usr/bin/python

# ---------------------------------------------------
# fetch command line arguments
# ---------------------------------------------------
import sys
index       = int(sys.argv[1])
cuda_id     = str(sys.argv[2])
folder      = str(sys.argv[3])
overwrite   = bool(int(sys.argv[4]))
# ---------------------------------------------------
#print('overwrite is set to ', overwrite)

# ---------------------------------------------------
# os parameters related to parallel execution:
# ---------------------------------------------------
import os

# filter out tf info messages,
#   for more info, see
#   https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
''' TF_CPP_MIN_LOG_LEVEL
0 = all messages are logged (default behavior)
1 = INFO messages are not printed
2 = INFO and WARNING messages are not printed
3 = INFO, WARNING, and ERROR messages are not printed
'''
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = cuda_id

# ---------------------------------------------------
# tensorflow parameters related to parallel execution:
# ---------------------------------------------------
import tensorflow as tf

# allow parallel execution with the following:
#   for more info, see [https://www.tensorflow.org/guide/gpu]
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)
# ---------------------------------------------------

# ---------------------------------------------------
# Imports
# ---------------------------------------------------
from . import utils
from . import models
import keras
import pandas as pd
import toml
import numpy as np
# ---------------------------------------------------

#print(f'running cfg file {index}...')

# ---------------------------------------------------
# Preparations
# ---------------------------------------------------
# Start a file manager
F = utils.files.FileManager(folder)
# Read the config file
config = toml.load(F.filename_of_config(index))
# ---------------------------------------------------

# ---------------------------------------------------
# Check if this model was already trained...
# ---------------------------------------------------
# if the file already exists and we don't want to overwrite it, don't start the training:
if os.path.isfile(F.filename_of_model(index)) and overwrite == False:
    #print(f'\t no training was done for {index} (model already exists and overwrite is set to False).')
    pass

# otherwise start the training:
else:
    #print(f'\t ...now starting the training for {index}...')
    # ---------------------------------------------------
    # Create the Dataset
    # ---------------------------------------------------
    train_ids = np.arange(config['file_ids']['train_ids'][0], config['file_ids']['train_ids'][-1]+1)
    val_ids = np.arange(config['file_ids']['val_ids'][0], config['file_ids']['val_ids'][-1]+1)
    test_ids = np.arange(config['file_ids']['test_ids'][0], config['file_ids']['test_ids'][-1]+1)

    dataset = utils.create_dataset.create_dataset(config['train_info']['data_path'],
                                                  train_ids,
                                                  val_ids,
                                                  test_ids[0:1], # don't load test files during training
                                                  lp_indices=config['bit_ids']['lp_ids'],
                                                  h_from_backcast = config['bit_ids']['h_from_backcast'])

    # ---------------------------------------------------
    # Clean up backend and set random seed
    # ---------------------------------------------------
    keras.backend.clear_session()
    utils.random_seeds.set_random_seeds(index)

    # ---------------------------------------------------
    # Select Model to be trained
    # ---------------------------------------------------
    model_id = config['train_info']['model_id']
    # infer the dimension of the input and output layer from the training data
    time_steps, output_dim = len(dataset['x_train'][0]), len(dataset['y_train'][0])
    # initialize the model
    M = models.ModelSelector(time_steps, output_dim, model_id,
                             filename_npzlog = F.filename_of_validation_npz(index),
                              y_val_length = dataset['y_val'].shape[0])
    # pass the dataset
    M.passdataset(config['train_info']['epochs'],
                    dataset['x_train'],
                    dataset['y_train'],
                    dataset['x_val'],
                    dataset['y_val'],
                    config['train_info']['batch_size'],
                    config['train_info']['batch_size_val'])

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------
    M.train_model(logfile=F.filename_of_training_progress(index))

    # ---------------------------------------------------
    # Evaluate
    # ---------------------------------------------------
    M.model.evaluate(dataset['x_val'], dataset['y_val'], batch_size=config['train_info']['batch_size_val'])
    # ---------------------------------------------------
    # Save history and model
    # ---------------------------------------------------

    # Save training info, such as runtime
    M.save_training_info(F.filename_of_training_info(index))

    # Save the model
    M.save_model(F.filename_of_model(index))
    # M.model.save(F.filename_of_model(index), overwrite=True, include_optimizer=False)

    # Save the training history
    M.save_training_history(F.filename_of_training_history(index))
    #df = pd.DataFrame({'loss':M.history.history['loss'], 'binary_accuracy':M.history.history['binary_accuracy']})
    #df.to_pickle(F.filename_of_training_history(index))
    M.save_validation_history(F.filename_of_validation_history(index))

    # ---------------------------------------------------
    # Clean up backend
    # ---------------------------------------------------
    keras.backend.clear_session()

    #print(f'\t finished training for {index}.')
# ---------------------------------------------------


