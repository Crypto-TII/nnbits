from . import models
from .metric import BitByBitAccuracy, bitbybitaccuracy
import numpy as np
import pandas as pd

class Network(object):
    def __init__(self,
                 data_train,
                 data_val,
                 selected_bits,
                 not_selected_bits,
                 model_id,
                 model_strength,
                 model_inputs,
                 model_outputs,
                 input_data_op='None',
                 predict_label=False,
                 epochs=10,
                 batchsize=4096,
                 verbose=False,
                 validation_batch_size=None):

        self.data_train = data_train
        self.data_val = data_val
        self.selected_bits = selected_bits
        self.not_selected_bits = not_selected_bits

        self.model_id = model_id
        self.model_strength = model_strength
        self.model_inputs = model_inputs
        self.model_outputs = model_outputs
        self.input_data_op = input_data_op
        self.predict_label = predict_label
        self.epochs = epochs
        self.batchsize = batchsize
        self.verbose = verbose
        if validation_batch_size is None:
            self.validation_batch_size = self.data_val.shape[0]
        else:
            self.validation_batch_size = validation_batch_size

        # # infer the number of input neurons
        # if 'remove' in self.input_data_op:
        #     self.N_INPUT_BITS = selected_bits.shape[1]
        # else:
        #     self.N_INPUT_BITS = data_train.shape[1]
        #     # if there is a label in the data, subtract one:
        #     if self.predict_label:
        #         self.N_INPUT_BITS = self.N_INPUT_BITS-1

        # # infer the number of output neurons
        # self.N_OUTPUT_BITS = not_selected_bits.shape[1]
        # if self.predict_label:
        #     self.N_OUTPUT_BITS = 1

        # initialization
        self.create_model()
        self.initial_weights = self.model.get_weights()

    def create_model(self):
        # --- model preparation
        constructor = getattr(models, self.model_id)
        self.model = constructor(self.model_inputs, self.model_outputs, self.model_strength)

        if self.verbose:
            self.model.compile(optimizer=self.model.optimizer,
                                loss=self.model.loss,
                                run_eagerly=self.model.run_eagerly,
                                metrics=self.model.metrics + ['acc'])

    def reset_weights(self):
        # from https://gist.github.com/jkleint/eb6dc49c861a1c21b612b568dd188668
        # alternative to complete re-initialization: shuffle the original weights
        weights = [np.random.permutation(w.flat).reshape(w.shape) for w in self.initial_weights]
        # Faster, but less random: only permutes along the first dimension
        # weights = [np.random.permutation(w) for w in weights]
        self.model.set_weights(weights)

    def create_nn_dataset(self, data, selection_id):
        import tensorflow as tf

        if self.predict_label:
            # gather only the bits until the second last one:
            x = data[:, :-1]
            label_position = self.data_train.shape[1]-1
            y = tf.gather(data, [label_position], axis=1)
        else:
            x = data.copy()
            y = tf.gather(data, self.selected_bits[selection_id], axis=1)

        if self.input_data_op == 'None':
            pass

        elif 'remove' in self.input_data_op:
            # removing all selected bits is equivalent to gathering all not selected bits
            x = tf.gather(x, self.not_selected_bits[selection_id], axis=1)

        elif 'zero' in self.input_data_op:
            # set the selected_bits to zero:
            x = x.copy()
            x[:, self.selected_bits[selection_id]] = 0
            x = tf.convert_to_tensor(x)

        return x, y

    def pass_bit_selections(self, selection_id):
        import tensorflow as tf

        N_TRAIN = self.data_train.shape[0]

        x, y = self.create_nn_dataset(self.data_train, selection_id)

        ds_train = tf.data.Dataset.from_tensor_slices((x, y))

        ds_train = ds_train.cache()
        ds_train = ds_train.shuffle(N_TRAIN)
        ds_train = ds_train.batch(self.batchsize)
        ds_train = ds_train.prefetch(tf.data.AUTOTUNE)

        self.ds_train = ds_train

    def pass_bit_selections_test(self, selection_id):
        import tensorflow as tf

        x, y = self.create_nn_dataset(self.data_val, selection_id)

        ds_test = tf.data.Dataset.from_tensor_slices((x, y))

        ds_test = ds_test.cache()
        ds_test = ds_test.batch(self.validation_batch_size)
        ds_test = ds_test.prefetch(tf.data.AUTOTUNE)

        self.ds_test = ds_test

    def train(self):

        if self.verbose:
            validation_data = self.ds_test
        else:
            validation_data = None

        history = self.model.fit(self.ds_train,
                                 epochs=self.epochs,
                                 verbose=self.verbose,
                                 validation_data=validation_data,
                                 callbacks=self.model.callbacks)
        self.df_history = pd.DataFrame(history.history)
        return history.history

    def test(self, filename):
        m = BitByBitAccuracy(self.model_outputs)
        for model_input, y_true in [next(iter(self.ds_test))]:
            y_pred = self.model.predict(model_input)
            m.update_state(y_true, y_pred)
        np.save(filename, m.accs.numpy())
        return 0
    
    def test_details(self, filename): 
        results = []
        for model_input, y_true in [next(iter(self.ds_test))]:
            y_pred = self.model.predict(model_input)
            result = bitbybitaccuracy(y_true, y_pred, threshold=0.5, filename=None, get_accuracy=False, reduce=False)
            results.append(result.numpy())
        np.save(filename, np.array(results).flatten())
        return 0
    
    def train_details(self, filename):
        results = []
        for model_input, y_true in [next(iter(self.ds_train))]:
            y_pred = self.model.predict(model_input)
            result = bitbybitaccuracy(y_true, y_pred, threshold=0.5, filename=None, get_accuracy=False, reduce=False)
            results.append(result.numpy())
        np.save(filename, np.array(results).flatten())
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