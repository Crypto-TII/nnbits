import datetime
import warnings
from tensorflow import keras
from . import utils

from nbeats_keras.model import NBeatsNet as NBeatsKeras
warnings.filterwarnings(action='ignore', message='Setting attributes')

metrics = [keras.metrics.BinaryAccuracy(threshold=0.5)]

class NBEATSModel:
    """
    This code follows an N-BEATS toy version of Philipp Eremy.

    Please see
    https://github.com/philipperemy/n-beats/blob/master/nbeats_keras/model.py.
    """

    def __init__(self,
                 epochs,
                 x_train,
                 y_train,
                 x_val,
                 y_val,
                 batch_size,
                 hidden_layer_units=64,
                 loss='Huber',
                 optimizer='sgd',
                 stack_types=(NBeatsKeras.GENERIC_BLOCK,
                              NBeatsKeras.GENERIC_BLOCK),
                 thetas_dim=(4, 4),
                 nb_blocks_per_stack=2,
                 share_weights_in_stack=True,
                 ):

        self.callbacks = []

        self.epochs = epochs
        self.batch_size = batch_size
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

        time_steps, output_dim = len(x_train[0]), len(y_train[0])

        utils.random_seeds.set_random_seeds()

        self.model = NBeatsKeras(backcast_length=time_steps,
                                 forecast_length=output_dim,
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

    def train_model(self, verbose=0):

        start = datetime.datetime.now()

        self.history = self.model.fit(self.x_train,
                                      self.y_train,
                                      validation_data=(self.x_val,
                                                       self.y_val),
                                      validation_batch_size=self.batch_size,
                                      epochs=self.epochs,
                                      batch_size=self.batch_size,
                                      callbacks=self.callbacks,
                                      verbose=verbose,
                                      shuffle=True,
                                      validation_freq=5)

        stop = datetime.datetime.now()

        self.total_seconds = (stop - start).total_seconds()