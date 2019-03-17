''' This module provides a universal location for defining possible
    values of different parameters.'''

possible_denoise_models = ['baseline', 'baseline_opt', 'unet', 'baseline_1e5_adam_mse', 'unet_1e5sgd_0.99_mse', 'unet_1e5sgd_0.99', 'unet_1e3sgd_0.9', 'unet_1e3sgd_0.99', 'unet_1e3sgd_0.99_mse', 'unet_1e3adam']
possible_desc_models = ['baseline', 'baseline_opt', 'baseline_opt_1e3sgd', 'baseline_opt_1e2', 'baseline_opt_1e1']
possible_suffixes = ['denoise', 'desc']
# A list of the optional arguments and their defaults
arg_list = {'model_denoise': 'baseline', 'model_desc': 'baseline', 'epochs_denoise': 10, 'epochs_desc': 10, 'nodisk': False, 'use_clean': False, 'denoise_suffix': None, 'desc_suffix': None}
