''' This module provides a universal location for defining possible
    values of different parameters.'''

possible_denoise_models = ['baseline', 'baselinemse', 'unet', 'dncnn', 'none']
possible_desc_models = ['baseline', 'baselinedog', 'baseline100', 'baseline250', 'baseline500']
possible_suffixes = ['denoise', 'desc']
# A list of the optional arguments and their defaults
arg_list = {'evaluate': False, 'pca': False, 'model_denoise': 'baseline', 'model_desc': 'baseline', 'epochs_denoise': 0, 'epochs_desc': 0, 'nodisk': False, 'use_clean': False, 'denoise_suffix': None, 'denoisertrain': None, 'optimizer_desc': 'default', 'optimizer_denoise': 'default', 'keep_results': False}
