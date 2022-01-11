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
print('overwrite is set to ', overwrite)

# ---------------------------------------------------
# os parameters related to parallel execution:
# ---------------------------------------------------
import os

# filter out tf info messages,
#   for more info, see
#   https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
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
import numpy as np
import keras
import toml
import nbeats_keras.model
# ---------------------------------------------------

print(f'running cfg file {index}...')

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
if os.path.isfile(F.filename_of_predictions(index)) and overwrite == False:
    print(f'\t no testing was done for {index} (prediction already exists and overwrite is set to False).')
    pass

# otherwise start the training:
else:
    print(f'\t ...now starting the testing for {index}...')
    # ---------------------------------------------------
    # Create the Dataset
    # ---------------------------------------------------
    dataset = utils.create_dataset.create_dataset(config['train_info']['data_path'],
                                                  config['file_ids']['train_ids'][0:1], # we don't use training during testing
                                                  config['file_ids']['val_ids'][0:1], # we don't use validation during testing
                                                  config['file_ids']['test_ids'],
                                                  lp_indices=config['bit_ids']['lp_ids'])

    # ---------------------------------------------------
    # Clean up backend and set random seed
    # ---------------------------------------------------
    keras.backend.clear_session()
    utils.random_seeds.set_random_seeds(42)

    # ---------------------------------------------------
    # Load the model to be tested
    # ---------------------------------------------------
    # (since we don't want to train, set compile=False)
    model = tf.keras.models.load_model(F.filename_of_model(index),
                                       compile=False)

    # ---------------------------------------------------
    # Test
    # ---------------------------------------------------
    y_pred = model.predict(dataset['x_test'])

    # y_pred is a floating point value.
    # The threshold for the decision if the outcome is 0 or 1 is 0.5:
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0

    # cast to a more performant data type
    y_pred = y_pred.astype('uint8')

    # reshape the array (removes last dimension)
    if y_pred.shape[-1] == 1:
        y_pred = y_pred.reshape(y_pred.shape[0:-1])

    # also save the actual values
    y_true = dataset['y_test']

    # ---------------------------------------------------
    # Save prediction
    # ---------------------------------------------------
    np.savez(F.filename_of_predictions(index), y_pred=y_pred, y_true=y_true)
    print(f'\t finished testing for {index}.')
# ---------------------------------------------------


