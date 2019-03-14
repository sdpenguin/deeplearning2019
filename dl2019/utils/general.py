''' General utilities for dl2019 '''

import tensorflow as tf
import numpy as np
import random

def set_random_seeds(value=1234):
    ''' Sets a range of random seeds to the same value.
        Good for reproducible experiments.
    '''
    tf.set_random_seed(value)
    np.random.seed(value)
    random.seed(value)