import warnings
from tensorflow import keras
import tensorflow as tf
from .. import utils
from ..gohr.train_nets import make_resnet, cyclic_lr
from .nbeatskerasconv import NBeatsKerasConv
import pandas as pd
# import pickle
# from keras.callbacks import LearningRateScheduler
import numpy as np
# import tensorflow.experimental.numpy as tnp
# # from tensorflow.keras import backend as K
# # from tensorflow.keras.layers import Concatenate
# # from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape
# # from tensorflow.keras.models import Model
# from tensorflow.keras.callbacks import EarlyStopping
# #from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
# from keras import backend as K
# from keras.regularizers import l2
from keras import backend
# import keras
# from keras.callbacks import Callback
import os
# from keras.utils import losses_utils
# from keras.utils import metrics_utils
# from keras.metrics import MeanRelativeError
from nbeats_keras.model import NBeatsNet as NBeatsKeras
import datetime
import time

warnings.filterwarnings(action='ignore', message='Setting attributes')

# Set filter for tensorflow info and warning messages:
#   for more info, see https://stackoverflow.com/questions/40426502/is-there-a-way-to-suppress-the-messages-tensorflow-prints
#   2 = INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# class NPLogger(Callback):
#     """Callback that streams epoch results to a CSV file.
#     Supports all values that can be represented as a string,
#     including 1D iterables such as `np.ndarray`.
#     Example:
#     ```python
#     csv_logger = CSVLogger('training.log')
#     model.fit(X_train, Y_train, callbacks=[csv_logger])
#     ```
#     Args:
#       filename: Filename of the CSV file, e.g. `'run/log.csv'`.
#       separator: String used to separate elements in the CSV file.
#       append: Boolean. True: append if file exists (useful for continuing
#           training). False: overwrite existing file.
#     """
#
#     def __init__(self, filename):
#         self.filename = filename
#         super(NPLogger, self).__init__()
#
#     def on_train_begin(self, logs=None):
#         pass
#
#     def on_epoch_end(self, epoch, logs=None):
#         #logs = logs or {}
#         result = logs['val_bitBybit_accuracy']
#         result = result.numpy()
#         np.savez(self.filename, bitbybit=result)

def bitBybit_accuracy(y_true, y_pred, threshold=0.5, filename=None, y_val_length=-1):
    #--- CAST y_pred to 0,1:
    # y_pred contains values which are not 0 or 1
    # based on the chosen threshold,
    # e.g. 0.5, a y_pred value of 0.25 is cast to 0,
    # while 0.75 is cast to 1.
    y_pred = tf.convert_to_tensor(y_pred)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_pred.dtype)
    """
    y_true = [[1,1,0],    [0,0,0], [1,1,1], [0,0,0]] 
    y_pred = [[1,0.75,0], [0,0,0], [1,1,1], [1,1,0]]
    result = [[ True,  True,  True],
               [ True,  True,  True],
               [ True,  True,  True],
               [False, False,  True]]
    """
    result = tf.math.equal(y_true, y_pred)
    # cast boolean to float 32:
    result = tf.cast(result, tf.float32)
    """
    [[1., 1., 1.],
    [1., 1., 1.],
    [1., 1., 1.],
    [0., 0., 1.]]
    --> 
    [0.75, 0.75, 1.]
    """
    result = backend.mean(result, axis=0)
    # TODO: Here, we aim to save the result ONLY if it is the validation result (not the training one).
    # Our current solution forces the user to choose a different batch size for validation and training.
    # Also, it only supports a single validation batch.
    if y_pred.shape[0] == y_val_length:
        np.savez(filename, X=result.numpy())
    return backend.max(result)

class BitbyBit(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='bitbybit_accuracy', dtype=None, threshold=0.5, filename=None, y_val_length=None):
        super(BitbyBit, self).__init__(
                    bitBybit_accuracy, name, dtype=dtype, threshold=threshold, filename=filename, y_val_length=y_val_length)

from nbeats_keras.model import NBeatsNet as NBeatsKeras

class NBEATSModel:
    """
    This code follows an N-BEATS toy version of Philipp Eremy.

    Please see
    https://github.com/philipperemy/n-beats/blob/master/nbeats_keras/model.py.
    """

    def __init__(self,
                 input_dim,
                 output_dim,
                 batch_size,
                 hidden_layer_units=64,
                 stack_types=(NBeatsKeras.GENERIC_BLOCK,
                              NBeatsKeras.GENERIC_BLOCK),
                 thetas_dim=(4, 4),
                 nb_blocks_per_stack=2,
                 share_weights_in_stack=True,
                 ):

        print('calling NBEATSModel')

        time_steps, output_dim = input_dim, output_dim

        self.model = NBeatsKeras(backcast_length=time_steps,
                                 forecast_length=output_dim,
                                 stack_types=stack_types,
                                 nb_blocks_per_stack=nb_blocks_per_stack,
                                 thetas_dim=thetas_dim,
                                 share_weights_in_stack=share_weights_in_stack,
                                 hidden_layer_units=hidden_layer_units)

        return self.model

class ModelSelector:
    """
    This class provides initialization of different models, as well as training, saving and loading instances for them.
    For N-BEATS, this code follows an N-BEATS toy version of Philipp Eremy.

    Please see
    https://github.com/philipperemy/n-beats/blob/master/nbeats_keras/model.py.
    """

    def __init__(self,
                 input_length,
                 output_length,
                 model_id,
                filename_npzlog=None,
                 y_val_length=-1):

        self.model_id = model_id

        utils.random_seeds.set_random_seeds(np.random.randint(1000))

        self.filename_npzlog = filename_npzlog
        self.y_val_length = y_val_length

        # ---------------------------------------------------
        # PARAMETER CHOICES
        # ---------------------------------------------------
        if model_id == 'nbeats_-1':
            # ---------------------------------------------------
            # test models
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic', 'generic']
            thetas_dim = [2, 1]
            nb_blocks_per_stack = 1
            hidden_layer_units = 32
            share_weights_in_stack = True

        elif model_id == 'nbeats_with_conv1d':
            # ---------------------------------------------------
            # instead of Gohr's network, try a simple generic stack and a simplified, Gohr-like conv1d stack
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic'] + ['conv1d_simple']
            thetas_dim = [2, 1]
            nb_blocks_per_stack = 1  #
            hidden_layer_units = 32
            share_weights_in_stack = False

        elif model_id == 'nbeats_0_quarter':
            # ---------------------------------------------------
            # "standard" model for SPECK 32/64 with 2**10 (1024) bits to be analyzed
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic'] * 3
            thetas_dim = [2] * 3
            nb_blocks_per_stack = 1
            hidden_layer_units = 16
            share_weights_in_stack = False

        elif model_id == 'nbeats_0_half':
            # ---------------------------------------------------
            # "standard" model for SPECK 32/64 with 2**10 (1024) bits to be analyzed
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic'] * 6
            thetas_dim = [2] * 6
            nb_blocks_per_stack = 2
            hidden_layer_units = 32
            share_weights_in_stack = False

        elif model_id == 'nbeats_0':
            # ---------------------------------------------------
            # "standard" model for SPECK 32/64 with 2**10 (1024) bits to be analyzed
            # ---------------------------------------------------

            optimizer = 'Adam'
            loss = 'Huber' #tf.keras.losses.BinaryCrossentropy(from_logits=True, axis=1) #
            stack_types = ['generic'] * 12
            thetas_dim = [2] * 12
            nb_blocks_per_stack = 4
            hidden_layer_units = 64
            share_weights_in_stack = False

        elif model_id == 'nbeats_0_with_conv1d':
            # ---------------------------------------------------
            # "standard" model for SPECK 32/64 with 2**10 (1024) bits to be analyzed
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic'] * 11 + ['conv1d_simple']
            thetas_dim = [2] * 11 + [1]
            nb_blocks_per_stack = 4
            hidden_layer_units = 64
            share_weights_in_stack = False

        elif model_id == 'nbeats_1':
            # ---------------------------------------------------
            # more powerful model for SPECK 64/128 with 2**12 (4096) bits to be analyzed
            # change compared to model_id 0 is the thetas_dim, which is increased to 8.
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic'] * 12
            thetas_dim = [8] * 12
            nb_blocks_per_stack = 4
            hidden_layer_units = 64
            share_weights_in_stack = False

        elif model_id == 'nbeats_2':
            # ---------------------------------------------------
            # more powerful model for SPECK 128/with 2**12 (16,384) bits to be analyzed
            # ---------------------------------------------------
            optimizer = 'Adam'
            loss = 'Huber'
            stack_types = ['generic'] * 12
            thetas_dim = [32] * 12
            nb_blocks_per_stack = 4
            hidden_layer_units = 256
            share_weights_in_stack = False

        # ---------------------------------------------------
        # CREATE AND COMPILE THE MODEL
        # ---------------------------------------------------
        #bitBybit_accuracy_metric = tf.keras.metrics.MeanMetricWrapper(fn=bitBybit_accuracy(filename='lognpz.npz'))
        # bitBybit_accuracy(filename='lognpz.npz')

        metrics = [keras.metrics.BinaryAccuracy(threshold=0.5),
                   BitbyBit(filename=self.filename_npzlog, y_val_length=self.y_val_length)]

        if ('nbeats' in model_id) and ('conv' in model_id):
            self.model = NBeatsKerasConv(backcast_length=input_length,
                                         forecast_length=output_length,
                                         stack_types=stack_types,
                                         nb_blocks_per_stack=nb_blocks_per_stack,
                                         thetas_dim=thetas_dim,
                                         share_weights_in_stack=share_weights_in_stack,
                                         hidden_layer_units=hidden_layer_units)
            # Definition of the objective function and the optimizer.
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=metrics,
                               run_eagerly=True)

        elif 'nbeats' in model_id:
            self.model = NBeatsKeras(backcast_length=input_length,
                                     forecast_length=output_length,
                                     stack_types=stack_types,
                                     nb_blocks_per_stack=nb_blocks_per_stack,
                                     thetas_dim=thetas_dim,
                                     share_weights_in_stack=share_weights_in_stack,
                                     hidden_layer_units=hidden_layer_units)
            # Definition of the objective function and the optimizer.
            self.model.compile(loss=loss,
                               optimizer=optimizer,
                               metrics=metrics,
                               run_eagerly=True)

        elif model_id == 'gohr':
            # parameters from gohr.train_nets
            depth = 1
            self.model = make_resnet(depth=depth, reg_param=10 ** -5)
            self.model.compile(optimizer='adam', loss='mse', metrics=metrics)#['acc'])

        elif model_id == 'sklearn_DecisionTreeClassifier':
            from sklearn.tree import DecisionTreeClassifier
            self.model = DecisionTreeClassifier()

        elif model_id == 'sklearn_ExtraTreeClassifier':
            from sklearn.tree import ExtraTreeClassifier
            self.model = ExtraTreeClassifier()

        elif model_id == 'sklearn_ExtraTreesClassifier':
            from sklearn.ensemble import ExtraTreesClassifier
            self.model = ExtraTreesClassifier()

        elif model_id == 'sklearn_KNeighborsClassifier':
            from sklearn.neighbors import KNeighborsClassifier
            self.model = KNeighborsClassifier()

        elif model_id == 'sklearn_MLPClassifier':
            from sklearn.neural_network import MLPClassifier
            self.model = MLPClassifier()

        elif model_id == 'sklearn_RadiusNeighborsClassifier':
            from sklearn.neighbors import RadiusNeighborsClassifier
            self.model = RadiusNeighborsClassifier()

        elif model_id == 'sklearn_RandomForestClassifier':
            from sklearn.ensemble import RandomForestClassifier
            self.model = RandomForestClassifier()
            print('model random forest classifier')

        elif model_id == 'sklearn_RidgeClassifier':
            from sklearn.linear_model import RidgeClassifier
            self.model = RidgeClassifier()

        elif model_id == 'sklearn_RidgeClassifierCV':
            from sklearn.linear_model import RidgeClassifierCV
            self.model = RidgeClassifierCV()

        if 'sklearn' not in model_id:
            self.initial_weights = self.model.get_weights()

    def set_filename_npzlog(self, filename):
        self.filename_npzlog = filename

    def reset_weights(self):
        # from https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
        # alternative to complete re-initialization: shuffle the original weights
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        self.model.set_weights(weights)

    def passdataset(self,
                    epochs,
                    x_train,
                    y_train,
                    x_val,
                    y_val,
                    batch_size,
                    val_batch_size=None):

        self.epochs = epochs
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val
        if val_batch_size is None:
            self.val_batch_size = batch_size
        else:
            self.val_batch_size = val_batch_size

        # if 'sklearn' in self.model_id and self.y_train.shape[-1]==1:
        #    import numpy as np
        #    self.y_train = np.ravel(self.y_train)
        #    self.y_val = np.ravel(self.y_val)

    def train_model(self, verbose=0, logfile=None):

        start = time.time()

        # self.callbacks = [CustomStopper(monitor='binary_accuracy',
        #                                 mode='max',
        #                                 patience=10,
        #                                 restore_best_weights=True,
        #                                 start_epoch=20)]


        self.callbacks = [] # [EarlyStopping(monitor='val_loss', mode='min', patience=35, restore_best_weights=True)] #[]#
        #self.callbacks = [EarlyStopping(monitor='binary_accuracy', mode='max', patience=10, restore_best_weights=True)]#, ToggleMetrics()]
        #self.callbacks = [EarlyStopping(monitor='val_mean_metric_wrapper', mode='max', patience=10, restore_best_weights=True)]

        if logfile:
            from keras.callbacks import CSVLogger
            csv_logger = CSVLogger(logfile, append=False, separator=';')
            self.callbacks += [csv_logger]

        if 'nbeats' in self.model_id:

            self.history = self.model.fit(self.x_train,
                                          self.y_train,
                                          validation_data=(self.x_val,
                                                           self.y_val),
                                          validation_batch_size=self.val_batch_size,
                                          epochs=self.epochs,
                                          batch_size=self.batch_size,
                                          callbacks=self.callbacks,
                                          verbose=verbose,
                                          shuffle=True,
                                          validation_freq=1)

            y_pred = self.model.predict(self.x_val, batch_size=self.val_batch_size)
            y_pred = y_pred.reshape(self.y_val.shape)
            bitBybit_accuracy(self.y_val, y_pred, filename=self.filename_npzlog, y_val_length=self.y_val_length)

        elif self.model_id == 'gohr':

            # create learning rate schedule
            lr = LearningRateScheduler(cyclic_lr(10, 0.002, 0.0001))
            # add the learning rate schedule to the callbacks
            self.callbacks += [lr]

            self.history = self.model.fit(self.x_train,
                                            self.y_train,
                                            epochs=self.epochs,
                                            batch_size=self.batch_size,
                                            validation_data=(self.x_val, self.y_val),
                                            verbose=verbose,
                                            callbacks=self.callbacks);

        elif 'sklearn' in self.model_id:
            self.model.fit(self.x_train, self.y_train)

        stop = time.time()

        self.total_seconds = (stop - start)

    def save_model(self, filename):

        if ('nbeats' in self.model_id) or ('gohr' in self.model_id):
            #self.model.save(filename, overwrite=True, include_optimizer=False)
            self.model.save_weights(filename)

        elif 'sklearn' in self.model_id:
            pickle.dump(self.model, open(filename, 'wb'))

    def save_training_info(self, filename):
        """
        Note: read training info with df = pd.read_csv(filename).
        """
        df = pd.DataFrame([{'training_time': self.total_seconds}])
        df.to_csv(filename)

    def save_training_history(self, filename):

        if ('nbeats' in self.model_id) or ('gohr' in self.model_id):
            training_keys = [key for key in self.history.history.keys() if 'val' not in key]
            df = pd.DataFrame({key: self.history.history[key] for key in training_keys})
            # df = pd.DataFrame({'loss': self.model.history.history['loss'],
            #                    'binary_accuracy': self.model.history.history['binary_accuracy'],
            #                    'bitbybit_accuracy': self.model.history.history['bitbybit_accuracy']})
            df.to_pickle(filename)

        # elif 'gohr' in self.model_id:
        #     df = pd.DataFrame({'loss': self.model.history.history['loss'], 'binary_accuracy': self.model.history.history['binary_accuracy']})
        #     df.to_pickle(filename)

        else:
            pass

    def save_validation_history(self, filename):

        if ('nbeats' in self.model_id) or ('gohr' in self.model_id):
            # Save the validation history
            try:
                val_keys = [key for key in self.history.history.keys() if 'val' in key]
                df = pd.DataFrame({key: self.history.history[key] for key in val_keys})
                # df = pd.DataFrame({'loss': self.model.history.history['loss'],
                #                    'binary_accuracy': self.model.history.history['binary_accuracy'],
                #                    'bitbybit_accuracy': self.model.history.history['bitbybit_accuracy']})
                df.to_pickle(filename)
                # df = pd.DataFrame({'val_loss': self.model.history.history['val_loss'],
                #                    'val_binary_accuracy': self.model.history.history['val_binary_accuracy'],
                #                    'val_bitbybit_accuracy': self.model.history.history['val_bitbybit_accuracy']})
                # df.to_pickle(filename)
            except:
                pass

        # elif 'gohr' in self.model_id:
        #     df = pd.DataFrame({'val_loss': self.model.history.history['val_loss'], 'val_acc': self.model.history.history['val_acc']})
        #     df.to_pickle(filename)

        else:
            pass

    def load_model(self, filename):
        if ('nbeats' in self.model_id) or ('gohr' in self.model_id):
            #return tf.keras.models.load_model(filename, compile=False)
            return self.model.load_weights(filename)
        elif 'sklearn' in self.model_id:
            return pickle.load(open(filename, 'rb'))