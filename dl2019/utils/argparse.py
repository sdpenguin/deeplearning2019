import argparse

from dl2019.utils.possibles import possible_denoise_models, possible_desc_models, arg_list
from dl2019.utils.json_parse import JobSpecGenerator 
from dl2019.models.load_opt import opt_key_decode

def parse_args():
    parser = argparse.ArgumentParser(description='The main file to run the models for the Deep Learning 2019 Coursework.')
    parser.add_argument('dir_hpatches', action='store', help='Base hpatches directory containing all hpatches data in the default format.')
    parser.add_argument('dir_dump', action='store', help='Directory to place DenoiseHPatchesImproved formatted hpatches data, weights and losses.')
    parser.add_argument('dir_ktd', action='store', help='The keras_triplet_descriptor respository directory.')
    parser.add_argument('-f', '--agenda-file', dest='agenda', default=None, action='store', help='Specify a file path containing a JSON specification for jobs to run. Please see README for spec details.')
    parser.add_argument('--model-denoise', dest='model_denoise', default=arg_list['model_denoise'], action='store', help='The model to run for the denoiser. Must be one of {}'.format(possible_denoise_models))
    parser.add_argument('--model-desc', dest='model_desc', default=arg_list['model_desc'], action='store', help='The model to run for the descriptor. Must be one of {}'.format(possible_desc_models))
    parser.add_argument('--epochs-denoise', dest='epochs_denoise', default=arg_list['epochs_denoise'], type=int, action='store', help='Number of epochs for the denoiser.')
    parser.add_argument('--epochs-desc', dest='epochs_desc', default=arg_list['epochs_desc'], type=int, action='store', help='Number of epochs for the descriptor.')
    parser.add_argument('--optimizer', dest='optimizer', default=arg_list['optimizer'], type=str, action='store', help='Optimizer code to specify the optimizers you want to use. Default will be loaded if not specified for the models.')
    parser.add_argument('--nodisk', dest='nodisk', default=arg_list['nodisk'], type=bool, action='store', help='Set this flag to avoid saving or loading HPatches Denoiser Generators from disk.\
                        The HPatches data will be regenerated from scratch and not saved after it is generated. This may be useful for low-RAM systems.')
    parser.add_argument('--use-clean', dest='use_clean', default=arg_list['use_clean'], type=bool, action='store', help='Train the descriptor model on clean data, instead of data denoised using the\
                        trained denoiser.')
    parser.add_argument('--denoise-suffix', dest='denoise_suffix', default=arg_list['denoise_suffix'], type=str, action='store', help='Optional suffix for the denoiser folder.')
    parser.add_argument('--desc-suffix', dest='desc_suffix', default=arg_list['desc_suffix'], type=str, action='store', help='Optional suffix for the descriptor folder.')
    parsed = parser.parse_args()
    paths = {'dir_hpatches': parsed.dir_hpatches, 'dir_dump': parsed.dir_dump, 'dir_ktd': parsed.dir_ktd}
    print('MODELS: Printing agenda of models to run')
    if parsed.agenda:
        specs = JobSpecGenerator(parsed.agenda)
        job_list = []
        for job in specs:
            arg_checks(job)
            job_list.append(job)
        return (paths, job_list)
    else:
        job = {}
        job['model_denoise'] = parsed.model_denoise
        job['model_desc'] = parsed.model_desc
        job['epochs_denoise'] = parsed.epochs_denoise
        job['epochs_desc'] = parsed.epochs_desc
        job['nodisk'] = parsed.nodisk
        job['use_clean'] = parsed.use_clean
        job['denoise_suffix'] = parsed.denoise_suffix
        job['desc_suffix'] = parsed.desc_suffix
        job['optimizer'] = parsed.optimizer
        arg_checks(job)
        return (paths, [job])


def arg_checks(parsed):
    ''' Does preliminary checks to assert that the arguments are ok. '''
    print('Denoise Model: {} (Epochs: {}), Desc Model: {} (Epochs: {})'.format(parsed['model_denoise'], parsed['epochs_denoise'], parsed['model_desc'], parsed['epochs_desc']))
    opt_key_decode(parsed['optimizer']) # WIll raise an error if the opt key is invalid
    if parsed['model_denoise'] not in possible_denoise_models:
        raise ValueError('Your denoise model must be one of {}. Please amend possible_denoise_models if you have created a model.'.format(possible_denoise_models))
    if parsed['model_desc'] not in possible_desc_models:
        raise ValueError('Your descriptor model must be one of {}. Please amend possible_desc_models if you have created a model.'.format(possible_desc_models))
    if parsed['epochs_denoise'] == 0 and not parsed['use_clean']:
        print('WARNING: You have not specified use_clean as True, but you are not running the denoiser. If you have not trained the denoiser your descriptor training and val data may be garbage.')

