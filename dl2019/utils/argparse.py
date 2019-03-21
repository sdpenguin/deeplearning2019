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
    parser.add_argument('-e', '--evaluate', dest='evaluate', default=False, action='store_true', help='Set this flag to run evaluation (verification/matching/retrieval) tests on the specified descriptor model (with or without denoiser depending on use-clean.')
    parser.add_argument('--pca', dest='pca', default=False, action='store', help='Use the pca_power_law for evaluation. This may take longer, but generates more information.')
    parser.add_argument('--model-denoise', dest='model_denoise', default=arg_list['model_denoise'], action='store', help='The model to run for the denoiser. Must be one of {}'.format(possible_denoise_models))
    parser.add_argument('--model-desc', dest='model_desc', default=arg_list['model_desc'], action='store', help='The model to run for the descriptor. Must be one of {}'.format(possible_desc_models))
    parser.add_argument('--epochs-denoise', dest='epochs_denoise', default=arg_list['epochs_denoise'], type=int, action='store', help='Number of epochs for the denoiser.')
    parser.add_argument('--epochs-desc', dest='epochs_desc', default=arg_list['epochs_desc'], type=int, action='store', help='Number of epochs for the descriptor.')
    parser.add_argument('--optimizer-desc', dest='optimizer_desc', default=arg_list['optimizer_desc'], type=str, action='store', help='Descriptor ptimizer code to specify the optimizers you want to use. Default will be loaded if not specified for the model.')
    parser.add_argument('--optimizer-denoise', dest='optimizer_denoise', default=arg_list['optimizer_denoise'], type=str, action='store', help='Denoiser ptimizer code to specify the optimizers you want to use. Default will be loaded if not specified for the model.')
    parser.add_argument('--nodisk', dest='nodisk', default=arg_list['nodisk'], action='store_true', help='Set this flag to avoid saving or loading HPatches Denoiser Generators from disk.\
                        The HPatches data will be regenerated from scratch and not saved after it is generated. This may be useful for low-RAM systems.')
    parser.add_argument('--use-clean', dest='use_clean', default=arg_list['use_clean'], action='store_true', help='Set this flag to train/evaluate the descriptor model on clean data, instead of data denoised using the\
                        trained denoiser.')
    parser.add_argument('--denoise-suffix', dest='denoise_suffix', default=arg_list['denoise_suffix'], type=str, action='store', help='Optional suffix for the denoiser folder.')
    parser.add_argument('--denoisertrain', dest='denoisertrain', default=arg_list['denoisertrain'], type=str, action='store', help='Suffix specifying which denoiser the descriptor should be trained on data denoised by. If none specified then the model parameters are used to deduce the denoiser. If the model parameters do not correspond to the denoiser model given, then the descriptor WILL NOT be trained FURTHER, but may be evaulated on the denoiser parameters.')
    parser.add_argument('--keep-results', dest='keep_results', default=False, action='store_true', help='Set this flag to keep the results for the evaluation if you are running it. Warning: these folders can be in the GBs.')
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
        job['evaluate'] = parsed.evaluate
        job['pca'] = parsed.pca
        job['model_denoise'] = parsed.model_denoise
        job['model_desc'] = parsed.model_desc
        job['epochs_denoise'] = parsed.epochs_denoise
        job['epochs_desc'] = parsed.epochs_desc
        job['nodisk'] = parsed.nodisk
        job['use_clean'] = parsed.use_clean
        job['denoise_suffix'] = parsed.denoise_suffix
        job['denoisertrain'] = parsed.denoisertrain
        job['optimizer_desc'] = parsed.optimizer_desc
        job['optimizer_denoise'] = parsed.optimizer_denoise
        job['keep_results'] = parsed.keep_results
        arg_checks(job)
        return (paths, [job])


def arg_checks(parsed):
    ''' Does preliminary checks to assert that the arguments are ok. '''
    print('Denoise Model: {} (Opt:{}) (Suff:{}) (Epochs: {}), Desc Model: {} (Opt:{}) (Suff:{}) (Epochs: {})'.format(parsed['model_denoise'], parsed['optimizer_denoise'], parsed['denoise_suffix'], parsed['epochs_denoise'], parsed['model_desc'], parsed['optimizer_desc'], parsed['denoisertrain'], parsed['epochs_desc']))
    opt_key_decode(parsed['optimizer_desc']) # Will raise an error if the opt key is invalid
    opt_key_decode(parsed['optimizer_denoise'])
    if parsed['model_denoise'] == 'none' and not parsed['epochs_denoise'] == 0:
        raise ValueError('If your denoise model is none it cannot be trained. Set epochs_denoise to 0.')
    if parsed['model_denoise'] not in possible_denoise_models:
        raise ValueError('Your denoise model must be one of {}. Please amend possible_denoise_models if you have created a model.'.format(possible_denoise_models))
    if parsed['model_desc'] not in possible_desc_models:
        raise ValueError('Your descriptor model must be one of {}. Please amend possible_desc_models if you have created a model.'.format(possible_desc_models))
    if parsed['epochs_denoise'] == 0 and not parsed['use_clean']:
        print('WARNING: You have not specified use_clean as True, but you are not running the denoiser. If you have not trained the denoiser your descriptor training and val data may be garbage.')
