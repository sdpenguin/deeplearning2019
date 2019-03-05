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

try:
    from keras_triplet_descriptor.read_data import HPatches, DataGeneratorDesc, hpatches_sequence_folder, tps
    # If an import fails here, you need to make utils import relatively
    from keras_triplet_descriptor.utils import generate_desc_csv, plot_denoise, plot_triplet
except ImportError as e:
    raise ImportError('{}\nYou may need to add the keras_triplet_descriptor directory to the PATH \
or run from the directory above it. You may also need to add an __init__.py file to the directory.'.format(e))

from dl2019.utils.datastats import data_stats
from dl2019.utils.hpatches import DenoiseHPatchesImproved
from dl2019.models.callback import SaveProgress
from dl2019.models.load import get_latest_epoch, get_denoise_model, get_descriptor_model

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
def train_denoise(seqs_val, seqs_train, dir_dump, model_type, epochs_denoise, nodisk):
    training_dir = os.path.join(dir_dump, model_type + '_denoise')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    # Denoise generator
    denoise_val = DenoiseHPatchesImproved(seqs_val, dump=dir_dump, suff='val', nodisk=nodisk)
    denoise_train = DenoiseHPatchesImproved(seqs_train, dump=dir_dump, suff='train', nodisk=nodisk)
    # Initialise Denoise Model
    shape = tuple(list(data_stats(denoise_val.get_images(0), request='data_shape')) + [1])
    denoise_model = get_denoise_model(shape, model_type)
    optimizer = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    denoise_model.compile(l='mean_absolute_error', o=optimizer, m=['mae'])
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
    elif epochs_denoise == 0:
        print('WARNING: The denoiser has not been previously trained and will not be trained\n\
this time. This may cause problems training on denoised data for the Descriptor.')
        time.sleep(3)
    # Run model
    denoise_model.fit_generator(generator=denoise_train, epochs=epochs_denoise-max_epoch, verbose=1, validation_data=denoise_val, callbacks=callbacks)

    return (shape, denoise_model)

#%%
def train_descriptor(dir_hpatches, dir_dump, model_type, shape, epochs_desc, denoise_model, train_fnames, test_fnames, use_clean):
    training_dir = os.path.join(dir_dump, model_type + '_desc')
    if not os.path.exists(training_dir):
        os.makedirs(training_dir)
    # Descriptor Generator (TODO: Optimisation)
    hPatches = HPatches(train_fnames=train_fnames, test_fnames=test_fnames,
                        denoise_model=denoise_model, use_clean=use_clean)
    desc_train = DataGeneratorDesc(*hPatches.read_image_file(dir_hpatches, train=1), num_triplets=100000)
    desc_val = DataGeneratorDesc(*hPatches.read_image_file(dir_hpatches, train=0), num_triplets=10000)
    # Initialise the Descriptor Model
    desc_model = get_descriptor_model(shape, model_type)
    optimizer = keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    desc_model.compile(loss='mean_absolute_error', optimizer=optimizer)
    # Existing Epochs
    max_epoch = get_latest_epoch(training_dir)
    # Keras callbacks
    callback_log = keras.callbacks.TensorBoard(log_dir='./logs/{}_desc'.format(model_type), histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
    callback_save = SaveProgress(training_dir, curr_epoch=max_epoch)
    callbacks = [callback_log, callback_save]
    # Load existing weights
    if max_epoch is not 0:
        desc_model.load_weights(os.path.join(training_dir, '{}.h5'.format(max_epoch)))
    # Run model
    desc_model.fit_generator(generator=desc_train, epochs=epochs_desc-max_epoch, verbose=1, validation_data=desc_val, callbacks=callbacks)

#%%
def main(dir_ktd, dir_hpatches, dir_dump, model_type, epochs_denoise, epochs_desc, use_clean, nodisk):
    (seqs_val, seqs_train, train_fnames, test_fnames) = walk_hpatches(dir_ktd, dir_hpatches)
    (shape, denoise_model) = train_denoise(seqs_val, seqs_train, dir_dump, model_type, epochs_denoise, nodisk)
    train_descriptor(dir_hpatches, dir_dump, model_type, shape, epochs_desc, denoise_model, train_fnames, test_fnames, use_clean)

if __name__=='__main__':
    # Specify a path to the Keras Triplet Descriptor Repository
    main(parsed.dir_ktd, parsed.dir_hpatches, parsed.dir_dump,
         parsed.model_type, parsed.epochs_denoise, parsed.epochs_desc,
         parsed.use_clean, parsed.nodisk)
