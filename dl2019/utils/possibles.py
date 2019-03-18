''' This module provides a universal location for defining possible
    values of different parameters.'''

possible_denoise_models = ['baseline', 'baselinemse', 'unet']
possible_desc_models = ['baseline']
possible_suffixes = ['denoise', 'desc']
# A list of the optional arguments and their defaults
arg_list = {'model_denoise': 'baseline', 'model_desc': 'baseline', 'epochs_denoise': 10, 'epochs_desc': 10, 'nodisk': False, 'use_clean': False, 'denoise_suffix': None, 'desc_suffix': None, 'optimizer': 'default'}
