import numpy as np
import tensorflow as tf
import random

def set_random_seeds(seed_value=0):
    # Set numpy pseudo-random generator at a fixed value
    np.random.seed(seed_value)

    # Set tensorflow pseudo-random generator at a fixed value
    tf.random.set_seed(seed_value)

    # Set random package seed value
    random.seed(seed_value)