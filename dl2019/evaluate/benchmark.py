''' Runs benchmarks on the latest epoch of a given model or models.
    Credit to MatchLab Imperial, keras_triplet_descriptor for most of the code.'''
import tqdm
import dill
import os
import time
import json
import shutil
import numpy as np

from keras_triplet_descriptor.read_data import hpatches_sequence_folder
from dl2019.evaluate.hpatches_benchmark.utils.hpatch import *
from dl2019.evaluate.hpatches_benchmark.utils.tasks import *
from dl2019.evaluate.hpatches_benchmark.utils.misc import *
#from keras_triplet_descriptor.hpatches_benchmark.utils.results import *
from dl2019.evaluate.results_methods import *
from dl2019.utils.hpatches import dogs

def gen_desc_array(desc_model, output_folder_name, seqs_test, dir_dump, denoise_model=None, use_clean=False, dog=False):
    w = 32
    bs = 128
    output_dir = os.path.abspath(dir_dump)
    if use_clean:
        noisy_patches = 0
        denoise_model = None
    else:
        noisy_patches = 1
    for seq_path in tqdm(seqs_test):
        seq = hpatches_sequence_folder(seq_path, noise=noisy_patches)

        path = os.path.join(dir_dump, os.path.join('eval', os.path.join(output_folder_name, os.path.split(seq.name)[-1])))
        if not os.path.exists(path):
            os.makedirs(path)
        for tp in tps:
            # Do not recreate existing CSV files
            if os.path.exists(os.path.join(path,tp+'.csv')):
                continue
            n_patches = 0
            for i, patch in enumerate(getattr(seq, tp)):
                n_patches += 1

            patches_for_net = np.zeros((n_patches, 32, 32, 1))
            for i, patch in enumerate(getattr(seq, tp)):
                patches_for_net[i, :, :, 0] = cv2.resize(patch[0:w, 0:w], (32,32))
            ###
            outs = []
            
            n_batches = int(n_patches / bs) + 1
            for batch_idx in range(n_batches):
                st = batch_idx * bs
                if batch_idx == n_batches - 1:
                    if (batch_idx + 1) * bs > n_patches:
                        end = n_patches
                    else:
                        end = (batch_idx + 1) * bs
                else:
                    end = (batch_idx + 1) * bs
                if st >= end:
                    continue
                data_a = patches_for_net[st: end, :, :, :].astype(np.float32)
                if denoise_model:
                    data_a = np.clip(denoise_model.predict(data_a).astype(int), 0, 255).astype(np.float32)

                # convert to 5 channel if dog
                if dog:
                    data_a = data_a / 255 # Rescale the data
                    data_a = dogs(data_a)

                # compute output
                out_a = desc_model.predict(x=data_a)
                outs.append(out_a.reshape(-1, 128))

            res_desc = np.concatenate(outs)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches, -1))
            np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=';', fmt='%10.5f')   # X is an array

def evaluate(dir_ktd, dir_dump, output_folder_name, pca_power_law=False):
    path = os.path.join(dir_dump, os.path.join('eval', output_folder_name))
    results_dir = os.path.join(dir_dump, os.path.join('results', output_folder_name))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    descr = load_descrs(path,dist='L2',sep=';') # TODO: Make dist a parameter

    with open(os.path.join(dir_ktd, "splits.json")) as f:
        splits = json.load(f)

    splt = splits['a']

    for t in ['verification', 'matching', 'retrieval']: # TODO: Possibly split these up and enable the user to run only 1
        res_path = os.path.join(results_dir, output_folder_name+"_"+t+"_"+splt['name']+".p")
        if os.path.exists(res_path):
            print("Results for the {}, {} tasks, split {}, already cached!".format(output_folder_name, t, splt['name']))
        else:
            res = methods[t](os.path.join(dir_ktd, 'tasks'),descr,splt)
            dill.dump(res, open(res_path, "wb"))

    # do the PCA/power-law evaluation if wanted
    if pca_power_law: # opts['--pcapl']!='no'
        print('>> Running evaluation for %s normalisation' % blue("pca/power-law"))
        compute_pcapl(descr,splt)
        for t in ['verification', 'matching', 'retrieval']:
            res_path = os.path.join(results_dir, output_folder_name+"_pcapl_"+t+"_"+splt['name']+".p")
            if os.path.exists(res_path):
                print("Results for the %s, %s task, split %s,PCA/PL already cached!" %\
                      (output_folder_name,t,splt['name']))
            else:
                res = methods[t](os.path.join(dir_ktd, 'tasks'),descr,splt)
                dill.dump(res, open(res_path, "wb"))

def results(output_folder_name, dir_dump, dir_ktd, pca_power_law=False, more_info=False):
    results_dir = os.path.join(dir_dump,'results',output_folder_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    descrs = [output_folder_name]
    with open(os.path.join(dir_ktd, "splits.json")) as f:
        splits = json.load(f)
    splt = splits['a'] # TODO: split 'a' is hardcoded in this function and the one above
    results_array = []
    for t in ['verification', 'matching', 'retrieval']:
        print("%s task results:" % (green(t.capitalize())))
        for desc in descrs:
            curr_result = results_methods[t](results_dir, desc,splt,more_info)
            results_array.append([t, curr_result])
            if pca_power_law:
                results_methods[t](results_dir,desc+'_pcapl',splt,more_info)
            print
        print
    if not os.path.exists(os.path.join(dir_dump, 'overall_results')):
        os.makedirs(os.path.join(dir_dump, 'overall_results'))
    try:
        with open(os.path.join(dir_dump, os.path.join('overall_results', output_folder_name + '.npy')), 'w+') as results_file:
            np.save(results_file, np.array(results_array))
    except TypeError:
        with open(os.path.join(dir_dump, os.path.join('overall_results', output_folder_name + '.npy')), 'w+b') as results_file:
            np.save(results_file, np.array(results_array))

def run_evaluations(desc_model, seqs_test, dir_dump, dir_ktd, descriptor_name, denoisereval, pca_power_law=False, denoise_model=None, use_clean=False, keep_results=False, dog=False):
    print('EVALUATING: Generating a descriptor array for desc Model: {} Clean: {}.'.format(descriptor_name, use_clean))
    output_folder_name = descriptor_name
    if use_clean:
        output_folder_name = output_folder_name + '_clean' # Denoisereval will be ignored anyway
    else:
        output_folder_name = output_folder_name + '_' + denoisereval

    if os.path.exists(os.path.join(dir_dump, os.path.join('overall_results', output_folder_name + '.npy'))):
        print('SKIPPING EVALUATION: Evaluation file exists in overall_results directory.')
        return

    gen_desc_array(desc_model, output_folder_name, seqs_test, dir_dump, denoise_model=denoise_model, use_clean=use_clean, dog=dog)
    evaluate(dir_ktd, dir_dump, output_folder_name, pca_power_law)
    results(output_folder_name, dir_dump, dir_ktd, pca_power_law)
    if not keep_results:
        # These folders are often in excess of 1GB
        shutil.rmtree(os.path.join(dir_dump, os.path.join('eval', output_folder_name)))
        shutil.rmtree(os.path.join(dir_dump, os.path.join('results', output_folder_name)))

def load_results(dir_dump, descriptor, denoisertrain, denoisereval, use_clean):
    ''' Loads npy file results that have been generated and saved in dir_dump/overall_results.
        If use_clean is True, then simply set model_denoise etc. to None when you call this function.'''
    file_name = descriptor
    if not denoisertrain:
        file_name = file_name + '_clean'
    else:
        file_name = file_name + '_' + denoisertrain
    if not denoisereval and not use_clean:
        file_name = file_name + '_none_default'
    elif use_clean:
        file_name = file_name + '_clean'
    else:
        file_name = file_name + '_' + denoisereval

    results = np.load(os.path.join(dir_dump, 'overall_results', file_name + '.npy'))
    return results
