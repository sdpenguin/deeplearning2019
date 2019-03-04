import argparse

def parse_args():
    possible_models = ['baseline']
    parser = argparse.ArgumentParser(description='The main file to run the models for the Deep Learning 2019 Coursework.')
    parser.add_argument('dir_hpatches', action='store', help='Base hpatches directory containing all hpatches data in the default format.')
    parser.add_argument('dir_dump', action='store', help='Directory to place DenoiseHPatchesImproved formatted hpatches data, weights and losses.')
    parser.add_argument('dir_ktd', action='store', help='The keras_triplet_descriptor respository directory.')
    parser.add_argument('--model-type', dest='model_type', default='baseline', action='store', help='The model to run. Must be one of {}'.format(possible_models))
    parser.add_argument('--epochs-denoise', dest='epochs_denoise', default=10, type=int, action='store', help='Number of epochs for the denoiser.')
    parser.add_argument('--epochs-desc', dest='epochs_desc', default=10, type=int, action='store', help='Number of epochs for the descriptor.')
    parser.add_argument('--nodisk', dest='nodisk', default=False, type=bool, action='store', help='Set this flag to avoid saving or loading HPatches Denoiser Generators from disk.\
                        The HPatches data will be regenerated from scratch and not saved after it is generated. This may be useful for low-RAM systems.')
    parser.add_argument('--use-clean', dest='use_clean', default=False, type=bool, action='store', help='Train the descriptor model on clean data, instead of data denoised using the\
                        trained denoiser.')
    parsed = parser.parse_args()
    if parsed.model_type not in possible_models:
        raise ValueError('Your model must be one of {}.'.format(possible_models))
    return parsed