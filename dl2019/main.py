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
import datetime
import traceback

from keras.models import Model

try:
    from keras_triplet_descriptor.read_data import HPatches, hpatches_sequence_folder, tps
    # If an import fails here, you need to make utils import relatively
    from keras_triplet_descriptor.utils import generate_desc_csv, plot_denoise, plot_triplet
except ImportError as e:
    raise ImportError('{}\nYou may need to add the keras_triplet_descriptor directory to the PATH \
or run from the directory above it. You may also need to add an __init__.py file to the directory.'.format(e))

from dl2019.utils.argparse import parse_args
from dl2019.utils.datastats import data_stats
from dl2019.utils.hpatches import DenoiseHPatchesImproved, DataGeneratorDescImproved
from dl2019.utils.general import set_random_seeds
from dl2019.models.load_opt import opt_key_decode
from dl2019.models.callback import SaveProgress
from dl2019.models.load import get_latest_epoch, get_denoise_model, get_descriptor_model
from dl2019.evaluate.benchmark import run_evaluations

#%%
def get_denoisertrain(denoisertrain, model_type_denoise, optimizer_denoise, denoise_suffix, use_clean):
    ''' Returns names of the training denoiser and the evaluation denoiser and a parameter called descriptor_training that specifies whether or not to train the descriptor.
        denoisertrain is the denoiser that the descriptor uses to denoise the images before training.
        denoisereval is the denoiser used for evaluating if evaluate is True. It is the denoiser to actually be loaded..
        descriptor_training will be True if the two are equal, but false if not.'''
    descriptor_training = True # Whether or not to train the descriptor later
    denoisereval = '{}_{}'.format(model_type_denoise, optimizer_denoise)
    if denoise_suffix:
        denoisereval = denoisereval + '_{}'.format(denoise_suffix)

    if not denoisertrain: # Auto generate the suffix from the denoiser model given if no denoiser training model specified
        if not use_clean:
            denoisertrain = '{}_{}'.format(model_type_denoise, optimizer_denoise)
            if denoise_suffix:
                denoisertrain = denoisertrain + '_{}'.format(denoise_suffix)
        else:
            denoisertrain = 'clean'
    else:
        if not use_clean:
            if denoisereval != denoisertrain: # You have specified a different denoiser that descriptor should be trained with
                descriptor_training = False # Do not train the descriptor, (but maybe evaluate it with the given denoiser)
    print('The denoisertrain value has been updated to {}.'.format(denoisertrain))

    return (descriptor_training, denoisertrain, denoisereval)

#%%
def walk_hpatches(dir_ktd, dir_hpatches):
    ''' Obtains information about the directories in the hpatches folder.
        Returns ``(seqs_val, seqs_train, train_fnames, test_fnames)``.'''
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
def get_training_dirs(dir_dump, model_type_denoise, denoise_suffix, model_type_desc, denoisertrain, optimizer_desc, optimizer_denoise):
    ''' Returns the training directories for the denoiser and descriptor as ``(dir_denoise, dir_desc, dir_eval)``.
        denoisertrain is the optional suffix supplied indicating the descriptor should be evaluated on data denoised using that model, rather than
        the denoiser to be trained and returned as dir_denoise.
        dir_denoise and dir_desc are the folder names within dir_dump of the descriptor and denoiser to be trained and used for evaluation.
        dir_desc is the folder name for training the descriptor which includes the denoisertrain suffix.'''
    # Denoise
    dir_denoise = os.path.join(dir_dump, model_type_denoise + '_denoise' + '_{}'.format(optimizer_denoise))
    if denoise_suffix:
        dir_denoise = dir_denoise + '_{}'.format(denoise_suffix)
    # Desc
    dir_desc = os.path.join(dir_dump, model_type_desc + '_desc' + '_{}'.format(optimizer_desc))
    if denoisertrain:
        dir_desc = dir_desc + '_{}'.format(denoisertrain)

    dir_eval = model_type_desc + '_desc' + '_{}'.format(optimizer_desc)
    if denoisertrain:
        dir_eval = dir_eval + '_{}'.format(denoisertrain)

    return (dir_denoise, dir_desc, dir_eval)

#%%
def get_denoise_generator(seqs_val, seqs_train, dir_dump, nodisk):
    ''' Returns generators for train and test data for training the denoiser as ``(denoise_val, denoise_train)``. '''
    denoise_val = DenoiseHPatchesImproved(seqs_val, dump=dir_dump, suff='val', nodisk=nodisk)
    denoise_train = DenoiseHPatchesImproved(seqs_train, dump=dir_dump, suff='train', nodisk=nodisk)
    return (denoise_val, denoise_train)

def get_denoise_mod(model_type, shape, training_dir, optimizer):
    ''' Returns ``(denoise_model, callbacks, max_epoch)`` where ``max_epoch`` is the maximum epoch of the descriptor that was loaded if one already existed in ``training_dir`` '''
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
    ''' Trains the denoiser model for ``epochs_denoise`` with the given callbacks, training and validation generators'''
    # Run model
    denoise_model.fit_generator(generator=denoise_train, epochs=epochs_denoise-max_epoch, verbose=1, validation_data=denoise_val, callbacks=callbacks)

#%%
def get_desc_generator(dir_hpatches, train_fnames, test_fnames, denoise_model, use_clean, dog, batch_size):
    ''' Gets the generator for train and test data for the descriptor model training as ``(desc_val, desc_train)``, denoised using denoise_model.
        If dog is true then the generators will generate data suitable for the DoG descriptor model (5 channels). '''
    hPatches = HPatches(train_fnames=train_fnames, test_fnames=test_fnames,
                    denoise_model=denoise_model, use_clean=use_clean)
    desc_train = DataGeneratorDescImproved(*hPatches.read_image_file(dir_hpatches, train=1), num_triplets=100000, dog=dog, batch_size=batch_size)
    desc_val = DataGeneratorDescImproved(*hPatches.read_image_file(dir_hpatches, train=0), num_triplets=10000, dog=dog, batch_size=batch_size)
    return (desc_val, desc_train)

def get_desc_mod(model_type, shape, training_dir, optimizer):
    ''' Returns ``(desc_model, callbacks, max_epoch)`` where ``max_epoch`` is the maximum epoch of the descriptor that was loaded if one already existed in ``training_dir`` '''
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
    ''' Trains the descriptor model sepcified with the given callbacks and denoiser model used to denoise the data, for ``epochs_desc`` with the given training and validation generators '''
    # Run model
    desc_model.fit_generator(generator=desc_train, epochs=epochs_desc-max_epoch, verbose=1, validation_data=desc_val, callbacks=callbacks)

#%%
def main(dir_ktd, dir_hpatches, dir_dump, evaluate, pca, optimizer_desc, optimizer_denoise, model_type_denoise, epochs_denoise, model_type_desc,
         epochs_desc, use_clean, nodisk, denoise_suffix=None, denoisertrain=None, denoise_val=None, denoise_train=None,
         desc_val=None, desc_train=None, keep_results=None, prev_batch_size=50):
    # Defaults
    dog = False
    batch_size = 50
    # Modifications based on model names
    if model_type_desc in ['baselinedog']: # Models that use the DoG method
        dog = True
    if model_type_desc in ['baseline100']:
        batch_size = 100
    elif model_type_desc in ['baseline250']:
        batch_size = 250
    elif model_type_desc in ['baseline500']:
        batch_size = 500

        
    (descriptor_training, denoisertrain, denoisereval) = get_denoisertrain(denoisertrain, model_type_denoise, optimizer_denoise, denoise_suffix, use_clean)
    (seqs_val, seqs_train, train_fnames, test_fnames) = walk_hpatches(dir_ktd, dir_hpatches)
    (dir_denoise, dir_desc, dir_eval) = get_training_dirs(dir_dump, model_type_denoise, denoise_suffix, model_type_desc, denoisertrain, optimizer_desc, optimizer_denoise)

    # Run and/or get denoiser
    if epochs_denoise > 0:
        import tensorflow as tf
        # Do not regenerate the generators every time
        (denoise_model, denoise_callbacks, max_epoch_denoise) = get_denoise_mod(model_type_denoise, (32,32,1), dir_denoise, optimizer_denoise)
        if epochs_denoise - max_epoch_denoise > 0:
            if not denoise_val or not denoise_train:
                (denoise_val, denoise_train) = get_denoise_generator(seqs_val, seqs_train, dir_dump, nodisk)
            print('RUNNING: denoise ({} Suffix:{} Optimizer:{}) up to {} epochs.'.format(model_type_denoise, denoise_suffix, optimizer_denoise, epochs_denoise))
            train_denoise(denoise_model, denoise_callbacks, max_epoch_denoise, epochs_denoise, denoise_val, denoise_train)
        else:
            print('SKIPPING COMPLETE: denoise ({} Suffix:{}) up to {} epochs.'.format(model_type_denoise, denoise_suffix, epochs_denoise))
    elif not use_clean and model_type_denoise != "none":
        (denoise_model, _, _) = get_denoise_mod(model_type_denoise, (32,32,1), dir_denoise, optimizer_denoise)
    else:
        denoise_model = None

    # Run and/or get descriptor
    if epochs_desc > 0:
        import tensorflow as tf # Fix 1.0 Val Loss
        # Do not regenerate the generators every time
        (desc_model, desc_callbacks, max_epoch_desc) = get_desc_mod(model_type_desc, (32,32,1), dir_desc, optimizer_desc)
        if not descriptor_training: # Descriptor is not trained if denoisertrain does not match denoisereval
            print("SKIPPING TRAINING: descriptor ({} Suffix:{} Optimizer:{}) up to {} epochs. Desc_suffix =/= denoise model parameters.".format(model_type_desc, denoisertrain, optimizer_desc, epochs_desc))
        elif epochs_desc - max_epoch_desc > 0:
            if not desc_val or not desc_train or batch_size != prev_batch_size:
                (desc_val, desc_train) = get_desc_generator(dir_hpatches, train_fnames, test_fnames, denoise_model, use_clean, dog, batch_size)
            print('RUNNING: descriptor ({} Suffix:{} Optimizer:{}) up to {} epochs with denoiser {}.'.format(model_type_desc, denoisertrain, optimizer_desc, epochs_desc, denoisertrain))
            train_descriptor(desc_model, desc_callbacks, max_epoch_desc, epochs_desc, desc_val, desc_train)
        else:
            print('SKIPPING COMPLETE: descriptor ({} Suffix:{} Optimizer:{}) up to {} epochs.'.format(model_type_desc, denoisertrain, optimizer_desc, epochs_desc))
    
    # Evaluate dir_dump/dir_eval (trained on data from denoisertrain) and dir_dump/dir_denoise
    if evaluate:
        single_input_desc_model = Model(inputs=desc_model.get_layer(index=3).get_input_at(0), outputs=desc_model.get_layer(index=3).get_output_at(0))
        run_evaluations(single_input_desc_model, seqs_val, dir_dump, dir_ktd, dir_eval, denoisereval, pca_power_law=pca, denoise_model=denoise_model, use_clean=use_clean, keep_results=keep_results, dog=dog)

    # Return generators and the batch sizes in order to avoid redundant reloading/generation of data between jobs
    return (denoise_val, denoise_train, desc_val, desc_train, prev_batch_size)

if __name__=='__main__':
    # Create the error log file
    if os.path.exists("./errors.log"): # Get rid of ye olde error file
        with open("./errors.log", "a+") as error_file:
            error_file.write("\n\n" + str(datetime.datetime.now()) + "\n\n")
    # Reduce tesnsorflow warning output (remove all but errors)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # Parse commandline arguments to get the paths and a list of jobs to run
    (paths, jobs) = parse_args()
    # Set initial values for generators and batch_size
    (denoise_val, denoise_train, desc_val, desc_train, prev_batch_size) = (None, None, None, None, 50)
    # Iterate over jobs, running each one with its parameters
    # Wrapped in try except clause to let subsequent jobs continue if a job is cancelled or fails due to an error
    for job in jobs:
        try:
            # We import tensorflow and run explicitly to prevent the strange problem of constant 1.0 Val Loss
            if 'desc_model' in locals() or 'desc_model' in globals():
                del desc_model
            import tensorflow as tf
            with tf.Session() as sess:
                print('SETTING UP: Denoise: {}:{}:{} (Epochs: {}), Desc: {}:{}:{} (Epochs: {})'.format(job['model_denoise'], job['optimizer_denoise'], job['denoise_suffix'], job['epochs_denoise'], job['model_desc'], job['optimizer_desc'], job['denoisertrain'], job['epochs_desc']))
                (denoise_val, denoise_train, desc_val, desc_train, prev_batch_size) = main(paths['dir_ktd'], paths['dir_hpatches'], paths['dir_dump'],
                                                                          job['evaluate'], job['pca'], job['optimizer_desc'], job['optimizer_denoise'], job['model_denoise'],
                                                                          job['epochs_denoise'], job['model_desc'], job['epochs_desc'],
                                                                          job['use_clean'], job['nodisk'], job['denoise_suffix'],
                                                                          job['denoisertrain'], denoise_val, denoise_train, desc_val, desc_train, job['keep_results'],
                                                                          prev_batch_size)
        except KeyboardInterrupt:
            print("\nCANCELLING: cancelling the current job due to Ctrl+C. Continuing run.\n")
        except BaseException as e:
            #raise e # Remove the comment for debugging purposes in the terminal
            print("\nEXCEPTION!!! Please see errors.log for details. Continuing run.\n")
            with open("./errors.log", 'a+') as error_file:
                traceback.print_exc(file=error_file)
                error_file.write(str(e) + "\n" + str(job) + "\n")
