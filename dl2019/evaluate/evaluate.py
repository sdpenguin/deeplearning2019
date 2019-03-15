import os
import numpy as np

from matplotlib import pyplot as plt

from dl2019.models.load import get_latest_epoch
from dl2019.utils.possibles import possible_models, possible_suffixes

def load_train_test(dir_dump, model_type, suffix, suffix2=None):
    """ Returns the train and test error as two arrays for a given model.
        Also returns the epochs as the third element."""
    if model_type not in possible_models:
        raise ValueError("The model_type must be from: {}".format(possible_models))
    elif suffix not in possible_suffixes:
        raise ValueError("The suffix must be from: {}".format(possible_suffixes))
    output_dir = os.path.join(dir_dump, '{}_{}'.format(model_type, suffix))
    if suffix2:
        # This is an optional suffix supplied as the parameter denoise_suffix or desc_suffix
        output_dir = output_dir + '_{}'.format(suffix2)
    (train_error, test_error, epochs) = ([], [], [])
    num_epochs = get_latest_epoch(output_dir)
    for i in range(1, num_epochs+1):
        if os.path.exists(os.path.join(output_dir, '{}.npy'.format(i))):
            epochs.append(i)
            [train_err_single, test_err_single] = np.load(os.path.join(output_dir, '{}.npy'.format(i)))
            train_error.append(train_err_single)
            test_error.append(test_err_single)
    return (train_error, test_error, epochs)

def make_plot(dir_dump, model_type, suffix, suffix2=None, max_epoch=100):
    ''' Adds a plot of the train and test data for the specified model up to the specified epoch. '''
    (train, test, epochs) = load_train_test(dir_dump, model_type, suffix, suffix2)
    print(np.min(test))
    label = '{}-{}'.format(model_type, suffix)
    if suffix2:
        label = label + '-{}'.format(suffix2)
    plt.plot(epochs[0:max_epoch], test[0:max_epoch], label='test: {}'.format(label))
    plt.plot(epochs[0:max_epoch], train[0:max_epoch], label='train: {}'.format(label))
