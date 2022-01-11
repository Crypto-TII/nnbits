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
from . import models
import keras
import pandas as pd
import toml
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
if os.path.isfile(F.filename_of_model(index)) and overwrite == False:
    print(f'\t no training was done for {index} (model already exists and overwrite is set to False).')
    pass

# otherwise start the training:
else:
    print(f'\t ...now starting the training for {index}...')
    # ---------------------------------------------------
    # Create the Dataset
    # ---------------------------------------------------
    dataset = utils.create_dataset.create_dataset(config['train_info']['data_path'],
                                                  config['file_ids']['train_ids'],
                                                  config['file_ids']['val_ids'],
                                                  config['file_ids']['test_ids'][0:1], # don't load test files during training
                                                  lp_indices=config['bit_ids']['lp_ids'])

    # ---------------------------------------------------
    # Clean up backend and set random seed
    # ---------------------------------------------------
    keras.backend.clear_session()
    utils.random_seeds.set_random_seeds(42)

    # ---------------------------------------------------
    # Select Model to be trained
    # ---------------------------------------------------
    model_id = 0

    if model_id == 0:

        M = models.NBEATSModel(config['train_info']['epochs'],
                                dataset['x_train'],
                                dataset['y_train'],
                                dataset['x_val'],
                                dataset['y_val'],
                                config['train_info']['batch_size'],
                                optimizer='Adam',
                                loss = 'Huber',
                                stack_types=('generic', 'generic', 'generic', 'generic',
                                             'generic', 'generic', 'generic', 'generic',
                                             'generic', 'generic', 'generic', 'generic'),
                                thetas_dim=(2, 2, 2, 2,
                                            2, 2, 2, 2,
                                            2, 2, 2, 2),
                                nb_blocks_per_stack=4,
                                hidden_layer_units = 64,
                                share_weights_in_stack=False, # turn off weight sharing as recommended in [Oreshkin2020]
                                )

    # ---------------------------------------------------
    # Train
    # ---------------------------------------------------
    M.train_model()

    # ---------------------------------------------------
    # Save history and model
    # ---------------------------------------------------
    df = pd.DataFrame({'loss':M.history.history['loss'], 'binary_accuracy':M.history.history['binary_accuracy']})
    df.to_pickle(F.filename_of_training_history(index))

    df = pd.DataFrame({'val_loss':M.history.history['val_loss'], 'val_binary_accuracy':M.history.history['val_binary_accuracy']})
    df.to_pickle(F.filename_of_validation_history(index))

    # Save the model
    M.model.save(F.filename_of_model(index), overwrite=True, include_optimizer=False)

    print(f'\t finished training for {index}.')
# ---------------------------------------------------


