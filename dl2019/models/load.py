import os
import numpy as np
import glob

from dl2019.models.baseline import BaselineDenoise, BaselineDescriptor, BaselineDenoiseOpt, BaselineDescriptorOpt
from dl2019.models.unet import UNetDenoise

from dl2019.models.custommods import *

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
    elif model_type == 'baseline_opt':
        return BaselineDenoiseOpt(shape)
    elif model_type == 'unet':
        return UNetDenoise(shape)
    else:
        raise NotImplementedError('The denoise model type "{}" does not exist.'.format(model_type))

def get_descriptor_model(shape, model_type):
    ''' Get a descriptor model based on the model type. '''
    if model_type == 'baseline':
        return BaselineDescriptor(shape)
    elif model_type == 'baseline_opt':
        return BaselineDescriptorOpt(shape)
    else:
        raise NotImplementedError('The descriptor model type "{}" does not exist.'.format(model_type))
