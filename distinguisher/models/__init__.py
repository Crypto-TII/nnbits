import datetime
import warnings
from tensorflow import keras
import tensorflow as tf
from .. import utils
from ..gohr.train_nets import make_resnet, cyclic_lr
import pandas as pd
import time
import pickle
from keras.callbacks import LearningRateScheduler
import numpy as np
import tensorflow as tf
import tensorflow.experimental.numpy as tnp
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Input, Dense, Lambda, Subtract, Add, Reshape
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping
from keras.layers import Dense, Conv1D, Input, Reshape, Permute, Add, Flatten, BatchNormalization, Activation
from keras import backend as K
from keras.regularizers import l2
from keras import backend
import keras
from keras.callbacks import Callback
import os
from keras.utils import losses_utils
from keras.utils import metrics_utils
from keras.metrics import MeanRelativeError
#from nbeats_keras.model import NBeatsNet as NBeatsKeras

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
    # data = np.load(filename)
    # if data['train_toggle'] == 1:
    #     np.savez(filename, train_toggle=0, X=data['X'])
    # else:
    #     np.savez(filename, train_toggle=1, X=result.numpy())
    if y_pred.shape[0] == y_val_length:
        np.savez(filename, X=result.numpy())
    return backend.max(result)

class BitbyBit(tf.keras.metrics.MeanMetricWrapper):
    def __init__(self, name='bitbybit_accuracy', dtype=None, threshold=0.5, filename=None, y_val_length=None):
        super(BitbyBit, self).__init__(
                    bitBybit_accuracy, name, dtype=dtype, threshold=threshold, filename=filename, y_val_length=y_val_length)

# def smape_loss(y_true, y_pred):
#     """
#     sMAPE loss as defined in "Appendix A" of
#     http://www.forecastingprinciples.com/files/pdf/Makridakia-The%20M3%20Competition.pdf
#     :return: Loss value
#     """
#     # mask=tf.where(y_true,1.,0.)
#     mask = tf.cast(y_true, tf.bool)
#     mask = tf.cast(mask, tf.float32)
#     sym_sum = tf.abs(y_true) + tf.abs(y_pred)
#     condition = tf.cast(sym_sum, tf.bool)
#     weights = tf.where(condition, 1. / (sym_sum + 1e-8), 0.0)
#     return 200 * tnp.nanmean(tf.abs(y_pred - y_true) * weights * mask)


class NBeatsKerasConv:
    GENERIC_BLOCK = 'generic'
    CONV1D_BLOCK = 'conv1d'
    TREND_BLOCK = 'trend'
    SEASONALITY_BLOCK = 'seasonality'
    BINARY_BLOCK = 'binary'

    _BACKCAST = 'backcast'
    _FORECAST = 'forecast'

    def __init__(self,
                 input_dim=1,
                 output_dim=1,
                 exo_dim=0,
                 backcast_length=10,
                 forecast_length=1,
                 stack_types=(TREND_BLOCK, SEASONALITY_BLOCK),
                 nb_blocks_per_stack=3,
                 thetas_dim=(4, 8),
                 share_weights_in_stack=False,
                 hidden_layer_units=256,
                 nb_harmonics=None):

        self.stack_types = stack_types
        self.nb_blocks_per_stack = nb_blocks_per_stack
        self.thetas_dim = thetas_dim
        self.units = hidden_layer_units
        self.share_weights_in_stack = share_weights_in_stack
        self.backcast_length = backcast_length
        self.forecast_length = forecast_length
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.exo_dim = exo_dim
        self.input_shape = (self.backcast_length, self.input_dim)
        self.exo_shape = (self.backcast_length, self.exo_dim)
        self.output_shape = (self.forecast_length, self.output_dim)
        self.weights = {}
        self.nb_harmonics = nb_harmonics
        assert len(self.stack_types) == len(self.thetas_dim)

        x = Input(shape=self.input_shape, name='input_variable')
        x_ = {}
        for k in range(self.input_dim):
            x_[k] = Lambda(lambda z: z[..., k])(x)
        e_ = {}
        if self.has_exog():
            e = Input(shape=self.exo_shape, name='exos_variables')
            for k in range(self.exo_dim):
                e_[k] = Lambda(lambda z: z[..., k])(e)
        else:
            e = None
        y_ = {}

        for stack_id in range(len(self.stack_types)):
            stack_type = self.stack_types[stack_id]
            nb_poly = self.thetas_dim[stack_id]
            for block_id in range(self.nb_blocks_per_stack):
                backcast, forecast = self.create_block(x_, e_, stack_id, block_id, stack_type, nb_poly)
                for k in range(self.input_dim):
                    x_[k] = Subtract()([x_[k], backcast[k]])
                    if stack_id == 0 and block_id == 0:
                        y_[k] = forecast[k]
                    else:
                        y_[k] = Add()([y_[k], forecast[k]])

        for k in range(self.input_dim):
            y_[k] = Reshape(target_shape=(self.forecast_length, 1))(y_[k])
            x_[k] = Reshape(target_shape=(self.backcast_length, 1))(x_[k])
        if self.input_dim > 1:
            y_ = Concatenate()([y_[ll] for ll in range(self.input_dim)])
            x_ = Concatenate()([x_[ll] for ll in range(self.input_dim)])
        else:
            y_ = y_[0]
            x_ = x_[0]

        if self.input_dim != self.output_dim:
            y_ = Dense(self.output_dim, activation='linear', name='reg_y')(y_)
            x_ = Dense(self.output_dim, activation='linear', name='reg_x')(x_)

        inputs_x = [x, e] if self.has_exog() else x
        n_beats_forecast = Model(inputs_x, y_, name=self._FORECAST)
        n_beats_backcast = Model(inputs_x, x_, name=self._BACKCAST)

        self.models = {model.name: model for model in [n_beats_backcast, n_beats_forecast]}
        self.cast_type = self._FORECAST

    def has_exog(self):
        # exo/exog is short for 'exogenous variable', i.e. any input
        # features other than the target time-series itself.
        return self.exo_dim > 0

    @staticmethod
    def load(filepath, custom_objects=None, compile=True):
        from tensorflow.keras.models import load_model
        return load_model(filepath, custom_objects, compile)

    def _r(self, layer_with_weights, stack_id):
        # mechanism to restore weights when block share the same weights.
        # only useful when share_weights_in_stack=True.
        if self.share_weights_in_stack:
            layer_name = layer_with_weights.name.split('/')[-1]
            try:
                reused_weights = self.weights[stack_id][layer_name]
                return reused_weights
            except KeyError:
                pass
            if stack_id not in self.weights:
                self.weights[stack_id] = {}
            self.weights[stack_id][layer_name] = layer_with_weights
        return layer_with_weights

    def create_block(self, x, e, stack_id, block_id, stack_type, nb_poly):
        # register weights (useful when share_weights_in_stack=True)
        def reg(layer):
            return self._r(layer, stack_id)

        # update name (useful when share_weights_in_stack=True)
        def n(layer_name):
            return '/'.join([str(stack_id), str(block_id), stack_type, layer_name])

        backcast_ = {}
        forecast_ = {}

        # ----------
        # MODIFICATION
        # purpose: don't use the regular dense layers if we want to work with a convolutional network
        # added the following line.
        if 'conv1d' not in stack_type:
            d1 = reg(Dense(self.units, activation='relu', name=n('d1')))
            d2 = reg(Dense(self.units, activation='relu', name=n('d2')))
            d3 = reg(Dense(self.units, activation='relu', name=n('d3')))
            d4 = reg(Dense(self.units, activation='relu', name=n('d4')))

        if stack_type == 'generic':
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='linear', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='linear', name=n('forecast')))

        # ----------
        # MODIFICATION
        # purpose: if we aim to model binary data between 0 and 1 a linear output activation is not suitable.
        # --> therefore we subsitute the output activation for the backcast and forecast by a sigmoid activation:
        elif 'conv1d' in stack_type:
            theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = reg(Dense(self.backcast_length, activation='sigmoid', name=n('backcast')))
            forecast = reg(Dense(self.forecast_length, activation='sigmoid', name=n('forecast')))
        # (end of the modification.)
        # ----------

        elif stack_type == 'trend':
            theta_f = theta_b = reg(Dense(nb_poly, activation='linear', use_bias=False, name=n('theta_f_b')))
            backcast = Lambda(trend_model, arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})
            forecast = Lambda(trend_model, arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                                      'forecast_length': self.forecast_length})

        else:  # 'seasonality'
            if self.nb_harmonics:
                theta_b = reg(Dense(self.nb_harmonics, activation='linear', use_bias=False, name=n('theta_b')))
            else:
                theta_b = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_b')))
            theta_f = reg(Dense(self.forecast_length, activation='linear', use_bias=False, name=n('theta_f')))
            backcast = Lambda(seasonality_model,
                              arguments={'is_forecast': False, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
            forecast = Lambda(seasonality_model,
                              arguments={'is_forecast': True, 'backcast_length': self.backcast_length,
                                         'forecast_length': self.forecast_length})
        for k in range(self.input_dim):
            if self.has_exog():
                d0 = Concatenate()([x[k]] + [e[ll] for ll in range(self.exo_dim)])
            else:
                d0 = x[k]
            # ----------
            # MODIFICATION
            # purpose: if we work with convolutional layers, the output is generated not by the dense layers d1,..., d4.
            # In the following two blocks the alternative ways to obtain the output are added.
            if stack_type == 'conv1d':
                # ---- parameters
                reg_param = 10 ** -5
                num_filters = 32
                ks = 3
                d1 = 64
                d2 = 64
                # ----.
                d0 = Reshape((4, 16), name=n('Reshape'))(x[k])
                perm = Permute((2, 1))(d0)
                conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
                conv0 = BatchNormalization()(conv0)
                conv0 = Activation('relu')(conv0)
                shortcut = conv0
                conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(shortcut);
                conv1 = BatchNormalization()(conv1);
                conv1 = Activation('relu')(conv1);
                conv2 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv1);
                conv2 = BatchNormalization()(conv2);
                conv2 = Activation('relu')(conv2);
                shortcut = Add()([shortcut, conv2])
                flat1 = Flatten()(shortcut)
                dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1);
                dense1 = BatchNormalization()(dense1);
                dense1 = Activation('relu')(dense1);
                dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
                dense2 = BatchNormalization()(dense2);
                dense2 = Activation('relu')(dense2);
                d4_ = dense2
            elif stack_type == 'conv1d_simple':
                # ---- parameters
                reg_param = 10 ** -5
                num_filters = 32
                ks = 2
                d1 = 64
                d2 = 64
                # ----.
                d0 = Reshape((4, -1), name=n('Reshape'))(x[k])
                perm = Permute((2, 1))(d0)
                conv0 = Conv1D(num_filters, kernel_size=1, padding='same', kernel_regularizer=l2(reg_param))(perm)
                conv0 = Activation('relu')(conv0)
                conv1 = Conv1D(num_filters, kernel_size=ks, padding='same', kernel_regularizer=l2(reg_param))(conv0);
                conv1 = BatchNormalization()(conv1);
                conv1 = Activation('relu')(conv1);
                flat1 = Flatten()(conv1)
                dense1 = Dense(d1, kernel_regularizer=l2(reg_param))(flat1);
                dense1 = BatchNormalization()(dense1);
                dense1 = Activation('relu')(dense1);
                dense2 = Dense(d2, kernel_regularizer=l2(reg_param))(dense1);
                dense2 = BatchNormalization()(dense2);
                dense2 = Activation('relu')(dense2);
                d4_ = dense2
            # (end of the modification.)
            # ----------
            else:
                d1_ = d1(d0)
                d2_ = d2(d1_)
                d3_ = d3(d2_)
                d4_ = d4(d3_)
            theta_f_ = theta_f(d4_)
            theta_b_ = theta_b(d4_)
            backcast_[k] = backcast(theta_b_)
            forecast_[k] = forecast(theta_f_)

        return backcast_, forecast_

    def __getattr__(self, name):
        # https://github.com/faif/python-patterns
        # model.predict() instead of model.n_beats.predict()
        # same for fit(), train_on_batch()...
        attr = getattr(self.models[self._FORECAST], name)

        if not callable(attr):
            return attr

        def wrapper(*args, **kwargs):
            cast_type = self._FORECAST
            if attr.__name__ == 'predict' and 'return_backcast' in kwargs and kwargs['return_backcast']:
                del kwargs['return_backcast']
                cast_type = self._BACKCAST
            return getattr(self.models[cast_type], attr.__name__)(*args, **kwargs)

        return wrapper


def linear_space(backcast_length, forecast_length, is_forecast=True):
    # ls = K.arange(-float(backcast_length), float(forecast_length), 1) / forecast_length
    # return ls[backcast_length:] if is_forecast else K.abs(K.reverse(ls[:backcast_length], axes=0))
    horizon = forecast_length if is_forecast else backcast_length
    return K.arange(0, horizon) / horizon


def seasonality_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.get_shape().as_list()[-1]
    p1, p2 = (p // 2, p // 2) if p % 2 == 0 else (p // 2, p // 2 + 1)
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    s1 = K.stack([K.cos(2 * np.pi * i * t) for i in range(p1)])
    s2 = K.stack([K.sin(2 * np.pi * i * t) for i in range(p2)])
    if p == 1:
        s = s2
    else:
        s = K.concatenate([s1, s2], axis=0)
    s = K.cast(s, np.float32)
    return K.dot(thetas, s)


def trend_model(thetas, backcast_length, forecast_length, is_forecast):
    p = thetas.shape[-1]
    t = linear_space(backcast_length, forecast_length, is_forecast=is_forecast)
    t = K.transpose(K.stack([t ** i for i in range(p)]))
    t = K.cast(t, np.float32)
    return K.dot(thetas, K.transpose(t))

class CustomStopper(keras.callbacks.EarlyStopping):
    def __init__(self, start_epoch = 100, **kwargs): # add argument for starting epoch
        super(CustomStopper, self).__init__()
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

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

        # ############### model checkpoint callback:
        # filepath = filepath[:-3] + '_bestvalweights.h5'
        #
        # self.model_checkpoint_callback = keras.callbacks.ModelCheckpoint(
        #     filepath=filepath,
        #     verbose=0,
        #     monitor='val_binary_accuracy',
        #     mode='max',
        #     save_best_only=True)
        #
        # self.callbacks.append(self.model_checkpoint_callback)
        # ############### ############### ###############

        time_steps, output_dim = input_dim, output_dim

        #BackendType = NBeatsKeras
        # self.model = BackendType(backcast_length=time_steps,
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