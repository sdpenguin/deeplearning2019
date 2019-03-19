import os
import numpy as np

from matplotlib import pyplot as plt

from dl2019.models.load import get_latest_epoch
from dl2019.utils.possibles import possible_denoise_models, possible_desc_models, possible_suffixes
from dl2019.models.load_opt import opt_key_decode

def load_train_test(dir_dump, model_type, suffix, optimizer, suffix2=None, override_checks=False):
    """ Returns the train and test error as two arrays for a given model.
        Also returns the epochs as the third element.
        If you want to override checks that the model is from a set of known values
        (e.g. if you have a non-customary folder name) then set override_checks to True."""
    if not override_checks:
        opt_key_decode(optimizer) # Check optimizer is a valid one
        if model_type not in possible_denoise_models and model_type not in possible_desc_models:
            raise ValueError("The model_type must be from: {}".format([possible_desc_models, possible_denoise_models]))
        elif suffix not in possible_suffixes:
            raise ValueError("The suffix must be from: {}".format(possible_suffixes))
    output_dir = os.path.join(dir_dump, '{}_{}_{}'.format(model_type, suffix, optimizer))
    if suffix2:
        # This is an optional suffix supplied as the parameter denoise_suffix or desc_suffix
        output_dir = output_dir + '_{}'.format(suffix2)
    (train_error, test_error, train_loss, test_loss, epochs) = ([], [], [], [], [])
    num_epochs = get_latest_epoch(output_dir)
    for i in range(1, num_epochs+1):
        if os.path.exists(os.path.join(output_dir, '{}.npy'.format(i))):
            epochs.append(i)
            curr_data = np.load(os.path.join(output_dir, '{}.npy'.format(i)))
            for item in curr_data:
                if item[0] == 'mean_absolute_error':
                    train_err_single = item[1]
                elif item[0] == 'val_mean_absolute_error':
                    test_err_single = item[1]
                elif item[0] == 'loss':
                    train_loss_single = item[1]
                elif item[0] == 'val_loss':
                    test_loss_single = item[1]
            train_error.append(train_err_single)
            test_error.append(test_err_single)
            train_loss.append(train_loss_single)
            test_loss.append(test_loss_single)
    return (train_error, test_error, train_loss, test_loss, epochs)

def make_plot(dir_dump, model_type, suffix, optimizer, suffix2=None, max_epoch=100, override_checks=False, mae=True):
    ''' Adds a plot of the train and test data for the specified model up to the specified epoch.
        If mae is False then the model-speicfic loss (which may be the mae) is plotted instead. This may not be comparable.'''
    (train_err, test_err, trainloss, testloss, epochs) = load_train_test(dir_dump, model_type, suffix, optimizer, suffix2, override_checks)
    if mae:
        train = train_err
        test = test_err
    else:
        train = trainloss
        test = testloss
    print(np.min(test))
    label = '{}-{}'.format(model_type, suffix)
    if suffix2:
        label = label + '-{}'.format(suffix2)
    plt.plot(epochs[0:max_epoch], test[0:max_epoch], label='test: {}'.format(label))
    plt.plot(epochs[0:max_epoch], train[0:max_epoch], label='train: {}'.format(label))
