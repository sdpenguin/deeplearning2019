#%%
# Global parameters
import argparse
parser = argparse.ArgumentParser(description='The main file to run the models for the Deep Learning 2019 Coursework.')
parser.add_argument('dir_hpatches', action='store')
parser.add_argument('dir_dump', action='store')
parser.add_argument('dir_ktd', action='store', help='The keras_triplet_descriptor respository directory.')
args = parser.parse_args()
dir_hpatches = args.dir_hpatches
dir_dump = args.dir_dump
dir_ktd = args.dir_ktd

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

# Specify a path to the Keras Triplet Descriptor Repository
dir_ktdgit = os.path.abspath(dir_ktd)
sys.path.insert(0, dir_ktdgit)
from read_data import HPatches, DataGeneratorDesc, hpatches_sequence_folder, tps
from utils import generate_desc_csv, plot_denoise, plot_triplet

from tfwaleed.datastats import *
from tfwaleed.hpatches import DenoiseHPatchesImproved
from models.baseline import BaselineDenoise, triplet_loss
from models.callback import SaveProgress
from models.load import get_latest_epoch


#%%
# Directories for training and testing
splits_file = os.path.join(dir_ktdgit, 'splits.json')
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
seqs_val = list(filter(lambda x: os.path.split(x)[-1] in split['test'], all_hpatches_dirs))


#%%
# Denoise generator
denoise_val = DenoiseHPatchesImproved(seqs_val, dump=dir_dump, suff='val', redump=False)
denoise_train = DenoiseHPatchesImproved(seqs_train, dump=dir_dump, suff='train', redump=False)


#%%
shape = tuple(list(data_stats(denoise_val.get_images(0), request='data_shape')) + [1])
denoise_model = BaselineDenoise(shape)
optimizer = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
denoise_model.compile(l='mean_absolute_error', o=optimizer, m=['mae'])


#%%
training_dir = os.path.join(dir_dump, 'baseline')
# Existing Epochs
max_epoch = get_latest_epoch(training_dir)
# Keras callbacks
callback_log = keras.callbacks.TensorBoard(log_dir='./logs/baseline', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None, update_freq='epoch')
callback_save = SaveProgress(training_dir, curr_epoch=max_epoch)
callbacks = [callback_log, callback_save]


#%%
# Load existing weights
if max_epoch is not 0:
    denoise_model.load_weights(os.path.join(training_dir, '{}.h5'.format(max_epoch)))
    
# Run model
denoise_model.fit_generator(generator=denoise_train, epochs=10-max_epoch, verbose=1, validation_data=denoise_val, callbacks=callbacks)
