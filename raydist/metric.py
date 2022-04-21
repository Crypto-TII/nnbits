import tensorflow as tf


def bitbybitaccuracy(y_true, y_pred, threshold=0.5, filename=None, get_accuracy=False):
    # --- CAST y_pred to 0,1:
    # y_pred contains values which are not 0 or 1
    # based on the chosen threshold,
    # e.g. 0.5, a y_pred value of 0.25 is cast to 0,
    # while 0.75 is cast to 1.
    # y_pred = tf.convert_to_tensor(y_pred)
    y_pred = y_pred.reshape(y_true.shape)
    threshold = tf.cast(threshold, y_pred.dtype)
    y_pred = tf.cast(y_pred > threshold, y_true.dtype)
    """
    y_true = [[1,1,0],    [0,0,0], [1,1,1], [0,0,0]] 
    y_pred = [[1,0.75,0], [0,0,0], [1,1,1], [1,1,0]]
    result = [[ True,  True,  True],
               [ True,  True,  True],
               [ True,  True,  True],
               [False, False,  True]]
    """
    result = tf.math.equal(y_true, y_pred)
    # cast boolean to int:
    result = tf.cast(result, tf.uint64)
    """
    sum up along axis 0 (column-wise)
    [[1, 1, 1],
    [1, 1, 1],
    [1, 1, 1],
    [0., 0, 1]]
    --> 
    [3, 3, 4]
    """
    result = tf.keras.backend.sum(result, axis=0)
    if get_accuracy:
        result = result / len(y_pred)
    return result


class BitByBitAccuracy(tf.keras.metrics.Metric):
    """Adapted from https://www.tensorflow.org/api_docs/python/tf/keras/metrics/Metric"""

    def __init__(self, N_LABELS, name='bitbybitaccuracy', **kwargs):
        super(BitByBitAccuracy, self).__init__(name=name, **kwargs)
        self.bitbybit = self.add_weight(name='tp', initializer='zeros')
        self.counter = tf.zeros(N_LABELS, dtype=tf.uint64)
        self.accs = tf.zeros(N_LABELS, dtype=tf.float32)
        self.y_pred_length = 0

    @tf.autograph.experimental.do_not_convert
    def update_state(self, y_true, y_pred):
        bitbybitacc = bitbybitaccuracy(y_true, y_pred)
        self.counter += bitbybitacc
        self.y_pred_length += len(y_pred)
        self.accs = self.counter / self.y_pred_length
        self.bitbybit = tf.reduce_max(self.accs)

    def result(self):
        return self.bitbybit