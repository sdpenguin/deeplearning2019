#%%
# IMPORTS
import keras
import tensorflow as tf
import numpy as np
import sys
import os
import glob
import time
import cv2
import random
import json
import argparse

from keras.models import Model

try:
    from keras_triplet_descriptor.read_data import HPatches, DataGeneratorDesc, hpatches_sequence_folder, tps
    # If an import fails here, you need to make utils import relatively
    from keras_triplet_descriptor.utils import generate_desc_csv, plot_denoise, plot_triplet
except ImportError as e:
    raise ImportError('{}\nYou may need to add the keras_triplet_descriptor directory to the PATH \
or run from the directory above it. You may also need to add an __init__.py file to the directory.'.format(e))

from dl2019.utils.argparse import parse_args
from dl2019.utils.datastats import data_stats
from dl2019.utils.hpatches import DenoiseHPatchesImproved
from dl2019.utils.general import set_random_seeds
from dl2019.models.load_opt import opt_key_decode
from dl2019.models.callback import SaveProgress
from dl2019.models.load import get_latest_epoch, get_denoise_model, get_descriptor_model
from dl2019.evaluate.benchmark import run_evaluations

#%%
def walk_hpatches(dir_ktd, dir_hpatches):
    ''' Obtains information about the directories in the hpatches folder. '''
    # Directories for training and testing
    splits_file = os.path.join(dir_ktd, 'splits.json')
    try:
        splits_json = json.load(open(splits_file, 'rb')) # May need to change this to 'r'
    except TypeError:
        splits_json = json.load(open(splits_file, 'r'))
    split = splits_json['a'] # One of several different splits we could have used

    # All directories in the hpatches folder
    all_hpatches_dirs = glob.glob(dir_hpatches+'/*')
    # Split the directories between train and test
    train_fnames = split['train']
    test_fnames = split['test']
    seqs_train = list(filter(lambda x: os.path.split(x)[-1] in train_fnames, all_hpatches_dirs)) 
    seqs_val = list(filter(lambda x: os.path.split(x)[-1] in test_fnames, all_hpatches_dirs))

    return (seqs_val, seqs_train, train_fnames, test_fnames)

#%%
def get_training_dirs(dir_dump, model_type_denoise, denoise_suffix, model_type_desc, desc_suffix, optimizer):
    ''' Returns the training directories for the denoiser and descriptor. '''
    # Denoise
    dir_denoise = os.path.join(dir_dump, model_type_denoise + '_denoise' + '_{}'.format(optimizer))
    if denoise_suffix:
        dir_denoise = dir_denoise + '_{}'.format(denoise_suffix)
    # Desc
    dir_desc = os.path.join(dir_dump, model_type_desc + '_desc' + '_{}'.format(optimizer))
    if desc_suffix:
        dir_desc = dir_desc + '_{}'.format(desc_suffix)
    return (dir_denoise, dir_desc)

#%%
def get_denoise_generator(seqs_val, seqs_train, dir_dump, nodisk):
    ''' Returns generators for train and test data for training the denoiser. '''
    denoise_val = DenoiseHPatchesImproved(seqs_val, dump=dir_dump, suff='val', nodisk=nodisk)
    denoise_train = DenoiseHPatchesImproved(seqs_train, dump=dir_dump, suff='train', nodisk=nodisk)
    return (denoise_val, denoise_train)

def get_denoise_mod(model_type, shape, training_dir, optimizer):
    ''' Returns a denoise model compiled with an (ADAM) optimiser as default. '''
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    # Initialise Denoise Model
    denoise_model = get_denoise_model(shape, model_type)
    denoise_model.compile(optimizer=opt_key_decode(optimizer, model_suffix='denoise'))
    # Existing Epochs
    max_epoch = get_latest_epoch(training_dir)
    # Keras callbacks
    callback_log = keras.callbacks.TensorBoard(log_dir='./logs/{}_denoise'.format(model_type), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callback_save = SaveProgress(training_dir, curr_epoch=max_epoch)
    callbacks = [callback_log, callback_save]
    # Load existing weights
    if max_epoch is not 0:
        print('INFO: Loading existing Denoise model at epoch {}.'.format(max_epoch))
        denoise_model.load_weights(os.path.join(training_dir, '{}.h5'.format(max_epoch)))
    return (denoise_model, callbacks, max_epoch)

def train_denoise(denoise_model, callbacks, max_epoch, epochs_denoise, denoise_val, denoise_train):
    ''' Trains a denoise model. '''
    # Run model
    denoise_model.fit_generator(generator=denoise_train, epochs=epochs_denoise-max_epoch, verbose=1, validation_data=denoise_val, callbacks=callbacks)

#%%
def get_desc_generator(dir_hpatches, train_fnames, test_fnames, denoise_model, use_clean):
    ''' Gets the generator for train and test data for the descriptor model training. '''
    hPatches = HPatches(train_fnames=train_fnames, test_fnames=test_fnames,
                    denoise_model=denoise_model, use_clean=use_clean)
    desc_train = DataGeneratorDesc(*hPatches.read_image_file(dir_hpatches, train=1), num_triplets=100000)
    desc_val = DataGeneratorDesc(*hPatches.read_image_file(dir_hpatches, train=0), num_triplets=10000)
    return (desc_val, desc_train)

def get_desc_mod(model_type, shape, training_dir, optimizer):
    ''' Returns a descriptor model. '''
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    desc_model = get_descriptor_model(shape, model_type)
    desc_model.compile(optimizer=opt_key_decode(optimizer, model_suffix='desc'))
    # Existing Epochs
    max_epoch = get_latest_epoch(training_dir)
    # Keras callbacks
    callback_log = keras.callbacks.TensorBoard(log_dir='./logs/{}_desc'.format(model_type), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callback_save = SaveProgress(training_dir, curr_epoch=max_epoch)
    callbacks = [callback_log, callback_save]
    # Load existing weights
    if max_epoch is not 0:
        print('INFO: Found existing desc model at epoch {}'.format(max_epoch))
        desc_model.load_weights(os.path.join(training_dir, '{}.h5'.format(max_epoch)))
    return (desc_model, callbacks, max_epoch)

def train_descriptor(desc_model, callbacks, max_epoch, epochs_desc, desc_val, desc_train):
    ''' Trains the descriptor. '''
    # Run model
    desc_model.fit_generator(generator=desc_train, epochs=epochs_desc-max_epoch, verbose=1, validation_data=desc_val, callbacks=callbacks)

#%%
def main(dir_ktd, dir_hpatches, dir_dump, evaluate, pca, optimizer, model_type_denoise, epochs_denoise, model_type_desc,
         epochs_desc, use_clean, nodisk, denoise_suffix=None, desc_suffix=None, denoise_val=None, denoise_train=None,
         desc_val=None, desc_train=None):
    (seqs_val, seqs_train, train_fnames, test_fnames) = walk_hpatches(dir_ktd, dir_hpatches)
    (dir_denoise, dir_desc) = get_training_dirs(dir_dump, model_type_denoise, denoise_suffix, model_type_desc, desc_suffix, optimizer)
    if epochs_denoise > 0:
        import tensorflow as tf
        # Do not regenerate the generators every time
        (denoise_model, denoise_callbacks, max_epoch_denoise) = get_denoise_mod(model_type_denoise, (32,32,1), dir_denoise, optimizer)
        if epochs_denoise - max_epoch_denoise > 0:
            if not denoise_val or not denoise_train:
                (denoise_val, denoise_train) = get_denoise_generator(seqs_val, seqs_train, dir_dump, nodisk)
            print('RUNNING: denoise ({} Suffix:{} Optimizer:{}) up to {} epochs.'.format(model_type_denoise, denoise_suffix, optimizer, epochs_denoise))
            train_denoise(denoise_model, denoise_callbacks, max_epoch_denoise, epochs_denoise, denoise_val, denoise_train)
        else:
            print('SKIPPING COMPLETE: denoise ({} Suffix:{}) up to {} epochs.'.format(model_type_denoise, denoise_suffix, epochs_denoise))
    elif not use_clean:
        denoise_model = get_denoise_mod(model_type_denoise, (32,32,1), dir_denoise, optimizer)
    else:
        denoise_model = None
    if epochs_desc > 0:
        import tensorflow as tf # Fix 1.0 Val Loss
        # Do not regenerate the generators every time
        (desc_model, desc_callbacks, max_epoch_desc) = get_desc_mod(model_type_desc, (32,32,1), dir_desc, optimizer)
        if epochs_desc - max_epoch_desc > 0:
            if not desc_val or not desc_train:
                (desc_val, desc_train) = get_desc_generator(dir_hpatches, train_fnames, test_fnames, denoise_model, use_clean)
            print('RUNNING: descriptor ({} Suffix:{} Optimizer:{}) up to {} epochs.'.format(model_type_desc, desc_suffix, optimizer, epochs_desc))
            train_descriptor(desc_model, desc_callbacks, max_epoch_desc, epochs_desc, desc_val, desc_train)
        else:
            print('SKIPPING COMPLETED: descriptor ({} Suffix:{} Optimizer:{}) up to {} epochs.'.format(model_type_desc, desc_suffix, optimizer, epochs_desc))
    if evaluate:
        single_input_desc_model = Model(inputs=desc_model.get_layer('sequential_1').get_input_at(0), outputs=desc_model.get_layer('sequential_1').get_output_at(0)) # TYH
        run_evaluations(single_input_desc_model, model_type_desc, optimizer, seqs_val, dir_dump, dir_ktd,
                        suffix=desc_suffix, pca_power_law=pca, denoise_model=denoise_model, use_clean=use_clean)

    return (denoise_val, denoise_train, desc_val, desc_train)

if __name__=='__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # Reduce tesnsorflow warning output
    (paths, jobs) = parse_args()
    # We import tensorflow and run explicitly to prevent the strange problem of constant 1.0 Val Loss
    (denoise_val, denoise_train, desc_val, desc_train) = (None, None, None, None)
    for job in jobs:
        import tensorflow as tf
        with tf.Session() as sess:
            print('SETTING UP: Denoise: {}:{}:{} (Epochs: {}), Desc: {}:{}:{} (Epochs: {})'.format(job['model_denoise'], job['optimizer'], job['denoise_suffix'], job['epochs_denoise'], job['model_desc'], job['optimizer'], job['desc_suffix'], job['epochs_desc']))
            (denoise_val, denoise_train, desc_val, desc_train) = main(paths['dir_ktd'], paths['dir_hpatches'], paths['dir_dump'],
                                                                      job['evaluate'], job['pca'], job['optimizer'], job['model_denoise'],
                                                                      job['epochs_denoise'], job['model_desc'], job['epochs_desc'],
                                                                      job['use_clean'], job['nodisk'], job['denoise_suffix'],
                                                                      job['desc_suffix'], denoise_val, denoise_train, desc_val, desc_train)
