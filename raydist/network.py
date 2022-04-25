from . import models
from .metric import BitByBitAccuracy
import numpy as np
import pandas as pd

class Network(object):
    def __init__(self,
                 data_train,
                 data_val,
                 input_filters,
                 output_filters,
                 model_id,
                 model_strength,
                 model_inputs,
                 model_outputs,
                 set_memory_growth=True,
                 data_strategy='remove',
                 epochs=10,
                 batchsize=4096):

        self.data_train = data_train
        self.data_val = data_val
        self.input_filters = input_filters
        self.output_filters = output_filters

        self.model_id = model_id
        self.model_strength = model_strength
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.data_strategy = data_strategy
        self.epochs = epochs
        self.batchsize = batchsize

        # inferred class constants
        if data_strategy in ['remove', 'zero_gohr']:
            self.N_INPUT_BITS = input_filters.shape[1]
        elif data_strategy == 'zero':
            self.N_INPUT_BITS = data_train.shape[1]
        self.N_OUTPUT_BITS = output_filters.shape[1]

        # initialization
        self.create_model()
        self.initial_weights = self.model.get_weights()

    def create_model(self):
        # --- model preparation
        constructor = getattr(models, self.model_id)
        self.model = constructor(self.model_inputs, self.model_outputs, self.model_strength)

    def reset_weights(self):
        # from https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
        # alternative to complete re-initialization: shuffle the original weights
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        self.model.set_weights(weights)

    def pass_filters(self, filter_id):
        import tensorflow as tf

        N_TRAIN = self.data_train.shape[0]

        if self.data_strategy == 'remove':
            x = tf.gather(self.data_train, self.input_filters[filter_id], axis=1)
        elif self.data_strategy == 'zero':
            x = self.data_train.copy()
            x[:, self.output_filters[filter_id]] = 0
            x = tf.convert_to_tensor(x)
        elif self.data_strategy == 'zero_gohr':
            x = self.data_train.copy()
            # gather only the bits until the second last one:
            x = x[:, :-1]
            # set the target bit of the input filter to zero
            x[:, self.input_filters[filter_id]] = 0
            x = tf.convert_to_tensor(x)

        y = tf.gather(self.data_train, self.output_filters[filter_id], axis=1)

        ds_train = tf.data.Dataset.from_tensor_slices((x, y))

        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(N_TRAIN)
        ds_train = ds_train.batch(self.batchsize)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        self.ds_train = ds_train

    def pass_filters_test(self, filter_id):
        import tensorflow as tf

        N_VAL = self.data_val.shape[0]

        if self.data_strategy == 'remove':
            x = tf.gather(self.data_val, self.input_filters[filter_id], axis=1)
        elif self.data_strategy == 'zero':
            x = self.data_val.copy()
            x[:, self.output_filters[filter_id]] = 0
            x = tf.convert_to_tensor(x)
        elif self.data_strategy == 'zero_gohr':
            x = self.data_val.copy()
            # gather only the bits until the second last one:
            x = x[:, :-1]
            # set the target bit of the input filter to zero
            x[:, self.input_filters[filter_id]] = 0
            x = tf.convert_to_tensor(x)

        y = tf.gather(self.data_val, self.output_filters[filter_id], axis=1)

        ds_test = tf.data.Dataset.from_tensor_slices((x, y))

        ds_test = ds_test.cache()
        ds_test = ds_test.batch(N_VAL)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        self.ds_test = ds_test

    def train(self):
        history = self.model.fit(self.ds_train, epochs=self.epochs,
                                 verbose=False,
                                 #validation_data=self.ds_test,
                                 callbacks=self.model.callbacks)
        self.df_history = pd.DataFrame(history.history)
        return history.history

    def test(self, filename):
        m = BitByBitAccuracy(self.N_OUTPUT_BITS)
        for model_input, y_true in [next(iter(self.ds_test))]:
            y_pred = self.model.predict(model_input)
            m.update_state(y_true, y_pred)
        np.save(filename, m.accs.numpy())
        return 0

    def save_history(self, filename):
        self.df_history.to_pickle(filename)

    def evaluate(self, x, y):
        self.model.evaluate(x, y, verbose=True)

    def get_weights(self):
        return self.model.get_weights()

    def set_weights(self, weights):
        self.model.set_weights(weights)

    def load_weights(self, h5_file):
        self.h5_file = h5_file
        self.model.load_weights(self.h5_file)

    def save_weights(self, h5_file):
        self.h5_file = h5_file
        self.model.save_weights(self.h5_file)
        return True