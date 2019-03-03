import pickle
from tqdm import tqdm
import os

from read_data import *

class DenoiseHPatchesImproved(DenoiseHPatches):
    """Class for loading an HPatches sequence from a sequence folder.
       Data can be presupplied for this class.
       """
    itr = tps
    def __init__(self, seqs, batch_size = 32, dump=None, redump=False, suff=''):
        ''' If you already have the NPatches data in a folder, then just
            supply it as a 3-tuple in presupplied_data. This speeds up
            initialization.
        '''
        self.batch_size = batch_size
        self.dim = (32, 32)
        self.n_channels = 1
        (self.all_paths, self.sequences, self.sequences_n) = get_denoiser_data(seqs, self.dim, self.n_channels, dump=dump, redump=redump, suff=suff)
        self.on_epoch_end()


def get_denoiser_data(all_hpatches_dirs, dim=(32,32), num_channels=1, dump=None, redump=False, suff=''):
    ''' This will generate and/or load the data needed for the Denoise Generator.
        Set dump to the place you want to store the files.
        Set redump to true to rewrite the files.
        suff could be 'train' OR 'test' to generate a different generator for each of
        train and test.'''
    if dump and not redump and os.path.exists(os.path.join(dump, "./denoise_generator_{}paths.dat".format(suff))):
        print('Opening existing data')
        with open(os.path.join(dump, "./denoise_generator_{}paths.dat".format(suff)), 'rb') as denoise_file:
            all_paths = pickle.load(denoise_file)
        with open(os.path.join(dump, "./denoise_generator_{}seq.dat".format(suff)), 'rb') as denoise_file:
            sequences = pickle.load(denoise_file)
        with open(os.path.join(dump, "./denoise_generator_{}seqn.dat".format(suff)), 'rb') as denoise_file:
            sequences_n = pickle.load(denoise_file)
    else:
        print('Generating new data')
        if redump and not dump:
            raise ValueError('Please supply the dump path if redump is True.')
        all_paths = []
        sequences = {}
        sequences_n = {}
        for base in tqdm(all_hpatches_dirs):
            for t in tps:
                # Get the clean and noisy images respectively (containing multiple patches)
                all_patches = cv2.imread(os.path.join(base, t + '.png'), 0)
                all_noisy_patches = cv2.imread(os.path.join(base, t + '_noise.png'), 0)
                N = int(all_patches.shape[0] / dim[0])
                # Split the arrays into the individual images (adding a dimension)
                patches = np.array(np.split(all_patches, N),
                                   dtype=np.uint8)
                noisy_patches = np.array(np.split(all_noisy_patches, N),
                                         dtype=np.uint8) # N is the number of items in the image
                for i in range(N):
                    path = os.path.join(base, t, str(i) + '.png')
                    all_paths.append(path)
                    sequences[path] = patches[i]
                    sequences_n[path] = noisy_patches[i]
        if dump:
            print('Dumping generated data')
            with open(os.path.join(dump, "./denoise_generator_{}paths.dat".format(suff)), 'w+b') as denoise_file:
                pickle.dump(all_paths, denoise_file)
            with open(os.path.join(dump, "./denoise_generator_{}seq.dat".format(suff)), 'w+b') as denoise_file:
                pickle.dump(sequences, denoise_file)
            with open(os.path.join(dump, "./denoise_generator_{}seqn.dat".format(suff)), 'w+b') as denoise_file:
                pickle.dump(sequences_n, denoise_file)
    print('Finished')
    return (all_paths,  sequences, sequences_n)