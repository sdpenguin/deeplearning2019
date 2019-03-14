import argparse

from dl2019.utils.possibles import possible_models

def parse_args():
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
    parser.add_argument('--desc-only', dest='desc_only', default=False, type=bool, action='store', help='Skip generator creation and training for the denoiser and only train\
                        the descriptor using clean data.')
    parser.add_argument('--denoise-suffix', dest='denoise_suffix', default=None, type=str, action='store', help='Optional suffix for the denoiser folder.')
    parser.add_argument('--desc-suffix', dest='desc_suffix', default=None, type=str, action='store', help='Optional suffix for the descriptor folder.')
    parsed = parser.parse_args()
    if parsed.model_type not in possible_models:
        raise ValueError('Your model must be one of {}.'.format(possible_models))
    if parsed.desc_only and not parsed.use_clean:
        print('WARNING: You have not specified use_clean as True, but desc_only is True. This may cause issues if you have not yet trained the denoiser.')
    return parsed