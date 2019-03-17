import keras.optimizers as optimizers

from dl2019.utils.possibles import possible_suffixes

''' Define your optimizer keys here. These will contribute to model folder names too. '''

def opt_key_decode(opt_key='default', model_suffix='denoise'):
    '''Translates an optimizer key to parameters to load an optimizer.'''
    if not model_suffix in possible_suffixes:
        raise ValueError('Model suffix must be one of {}.'.format(possible_suffixes))
    if opt_key == 'default':
        return None # The model should implement a default optimizer
    elif opt_key == 'base':
        if model_suffix == 'denoise':
            return optimizers.sgd(lr=1e-5, momentum=0.9, nesterov=True)
        elif model_suffix == 'desc':
            return optimizers.sgd(lr=0.1)
    elif opt_key == 'base_opt':
        if model_suffix == 'denoise':
            return optimizers.Adam(lr=1e-5)
        elif model_suffix == 'desc':
            return optimizers.Adam(lr=1e-3)
    elif opt_key == 'base1e3':
        return optimizers.sgd(lr=1e-3, momentum=0.9, nesterov=True)
    elif opt_key == 'base1e30.99':
        return optimizers.sgd(lr=1e-3, momentum=0.99, nesterov=True)
    elif opt_key == 'base0.99':
        return optimizers.sgd(lr=1e-5, momentum=0.99, nesterov=True)
    else:
        raise ValueError('The optimizer key {} does not exist.'.format(opt_key))

