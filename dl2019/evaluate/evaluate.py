import os
import numpy as np

from dl2019.models.load import get_latest_epoch
from dl2019.utils.possibles import possible_models, possible_suffixes

def load_train_test(dir_dump, model_type, suffix):
    """ Returns the train and test error as two arrays for a given model.
        Also returns the epochs as the third element."""
    if model_type not in possible_models:
        raise ValueError("The model_type must be from: {}".format(possible_models))
    elif suffix not in possible_suffixes:
        raise ValueError("The suffix must be from: {}".format(possible_suffixes))
    output_dir = os.path.join(dir_dump, '{}_{}'.format(model_type, suffix))
    (train_error, test_error, epochs) = ([], [], [])
    num_epochs = get_latest_epoch(output_dir)
    for i in range(1, num_epochs+1):
        if os.path.exists(os.path.join(output_dir, '{}.npy'.format(i))):
            epochs.append(i)
            [train_err_single, test_err_single] = np.load(os.path.join(output_dir, '{}.npy'.format(i)))
            train_error.append(train_err_single)
            test_error.append(test_err_single)
    return (train_error, test_error, epochs)
