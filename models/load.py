import os
import numpy as np
import glob

from models.baseline import BaselineDenoise, BaselineDescriptor

def get_latest_epoch(dir='.'):
    ''' Gets the number of the latest epoch in dir as determined by the files.'''
    directory = os.path.join(dir, '*.h5')
    files = glob.glob(directory)
    if len(files) is not 0:
        file_numbers = [int(os.path.split(os.path.splitext(x)[0])[-1]) for x in files]
        return np.max(file_numbers)
    else:
        return 0

def get_denoise_model(shape, model_type):
    ''' Get a denoise model based on the model type. '''
    if model_type == 'baseline':
        return BaselineDenoise(shape)
    else:
        raise NotImplementedError('The denoise model type "{}" does not exist.'.format(model_type))

def get_descriptor_model(shape, model_type):
    ''' Get a descriptor model based on the model type. '''
    if model_type == 'baseline':
        return BaselineDescriptor(shape)
    else:
        raise NotImplementedError('The descriptor model type "{}" does not exist.'.format(model_type))
