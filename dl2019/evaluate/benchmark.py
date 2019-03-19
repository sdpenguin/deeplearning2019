''' Runs benchmarks on the latest epoch of a given model or models.
    Credit to MatchLab Imperial, keras_triplet_descriptor for most of the code.'''
import tqdm
import dill
import os
import time
import json

from keras_triplet_descriptor.read_data import hpatches_sequence_folder
from dl2019.evaluate.hpatches_benchmark.utils.hpatch import *
from dl2019.evaluate.hpatches_benchmark.utils.tasks import *
from dl2019.evaluate.hpatches_benchmark.utils.misc import *
#from keras_triplet_descriptor.hpatches_benchmark.utils.results import *
from dl2019.evaluate.results_methods import *

def gen_desc_array(desc_model, model_desc, model_denoise, optimizer_desc, optimizer_denoise, desc_suffix, denoise_suffix, seqs_test, dir_dump, denoise_model=None, use_clean=False):
    w = 32
    bs = 128
    output_dir = os.path.abspath(dir_dump)
    if use_clean:
        noisy_patches = 0
        denoise_model = None
    else:
        noisy_patches = 1
    file_name = model_desc + '_desc_' + optimizer_desc
    if not use_clean:
        if desc_suffix:
            file_name = file_name + '_{}'.format(desc_suffix)
        file_name = file_name + '--' + model_denoise + '_denoise_' + optimizer_denoise
    else:
        file_name = file_name + '_clean'
    if denoise_suffix:
        file_name = file_name + '_{}'.format(denoise_suffix)
    for seq_path in tqdm(seqs_test):
        seq = hpatches_sequence_folder(seq_path, noise=noisy_patches)

        path = os.path.join(dir_dump, os.path.join('eval', os.path.join(file_name, os.path.split(seq.name)[-1])))
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

                # compute output
                out_a = desc_model.predict(x=data_a)
                outs.append(out_a.reshape(-1, 128))

            res_desc = np.concatenate(outs)
            res_desc = np.reshape(res_desc, (n_patches, -1))
            out = np.reshape(res_desc, (n_patches, -1))
            np.savetxt(os.path.join(path,tp+'.csv'), out, delimiter=';', fmt='%10.5f')   # X is an array
    return file_name

def evaluate(dir_ktd, dir_dump, desc_name, pca_power_law=False):
    path = os.path.join(dir_dump, os.path.join('eval', desc_name))
    results_dir = os.path.join(dir_dump, os.path.join('results', desc_name))
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    descr = load_descrs(path,dist='L2',sep=';') # TODO: Make dist a parameter

    with open(os.path.join(dir_ktd, "splits.json")) as f:
        splits = json.load(f)

    splt = splits['a']

    for t in ['verification', 'matching', 'retrieval']: # TODO: Possibly split these up and enable the user to run only 1
        res_path = os.path.join(results_dir, desc_name+"_"+t+"_"+splt['name']+".p")
        if os.path.exists(res_path):
            print("Results for the {}, {} tasks, split {}, already cached!".format(desc_name, t, splt['name']))
        else:
            res = methods[t](os.path.join(dir_ktd, 'tasks'),descr,splt)
            dill.dump(res, open(res_path, "wb"))

    # do the PCA/power-law evaluation if wanted
    if pca_power_law: # opts['--pcapl']!='no'
        print('>> Running evaluation for %s normalisation' % blue("pca/power-law"))
        compute_pcapl(descr,splt)
        for t in ['verification', 'matching', 'retrieval']:
            res_path = os.path.join(results_dir, desc_name+"_pcapl_"+t+"_"+splt['name']+".p")
            if os.path.exists(res_path):
                print("Results for the %s, %s task, split %s,PCA/PL already cached!" %\
                      (desc_name,t,splt['name']))
            else:
                res = methods[t](os.path.join(dir_ktd, 'tasks'),descr,splt)
                dill.dump(res, open(res_path, "wb"))

def results(desc_name, dir_dump, dir_ktd, pca_power_law=False, more_info=False):
    results_dir = os.path.join(dir_dump,'results',desc_name)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    descrs = [desc_name]
    with open(os.path.join(dir_ktd, "splits.json")) as f:
        splits = json.load(f)
    splt = splits['a'] # TODO: split 'a' is hardcoded in this function and the one above
    for t in ['verification', 'matching', 'retrieval']:
        print("%s task results:" % (green(t.capitalize())))
        for desc in descrs:
            results_methods[t](results_dir, desc,splt,more_info)
            if pca_power_law:
                results_methods[t](results_dir,desc+'_pcapl',splt,more_info)
            print
        print

def run_evaluations(desc_model, model_desc, model_denoise, optimizer_desc, optimizer_denoise, seqs_test, dir_dump, dir_ktd, desc_suffix=None, denoise_suffix=None, pca_power_law=False, denoise_model=None, use_clean=False):
    # Generate and save a descriptor array
    # TODO: Print the name of the denoiser too
    print('EVALUATING: Generating a descriptor array for desc Model: {} Opt: {} Suffix: {} Clean: {}.'.format(model_desc, optimizer_desc, desc_suffix, use_clean))
    if not use_clean:
        print('denoise Model: {} Opt: {} Suffix: {}'.format(model_denoise, optimizer_denoise, denoise_suffix))
    desc_name = gen_desc_array(desc_model, model_desc, model_denoise, optimizer_desc, optimizer_denoise, desc_suffix, denoise_suffix, seqs_test, dir_dump, denoise_model=denoise_model, use_clean=False)
    evaluate(dir_ktd, dir_dump, desc_name, pca_power_law)
    results(desc_name, dir_dump, dir_ktd, pca_power_law)
