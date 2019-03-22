import pickle
from tqdm import tqdm
import os
import cv2

from skimage import filters

from keras_triplet_descriptor.read_data import *

def dogs(x, k=1.6):
    ''' Converts an single channel image array to a 5 channel array (original + 4 DoGs).
        k is effectively the bandpass coefficient. It specifies the width of the bandpass Gaussian filter.'''
    returnable_array = []
    for img in x:
        img_plus_dogs = img
        for idx,sigma in enumerate([4.0,8.0,16.0,32.0]):
            s1 = filters.gaussian(img,k*sigma)
            s2 = filters.gaussian(img,sigma)
            # Multiply by sigma to get scale invariance
            dog = s1 - s2
            # Append to array
            np.append(img_plus_dogs, dog, axis=-1)
        # Append set of images to overall batch
        returnable_array.append(img_plus_dogs)
    return np.array(returnable_array)



class DataGeneratorDescImproved(keras.utils.Sequence):
    # 'Generates data for Keras'
    def __init__(self, data, labels, num_triplets = 1000000, batch_size=50, dim=(32,32), n_channels=1, shuffle=True, dog=False):
        # 'Initialization'
        self.dog = dog
        self.transform = None
        self.out_triplets = True
        self.dim = dim
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.data = data
        self.labels = labels
        self.num_triplets = num_triplets
        self.on_epoch_end()

    def get_image(self, t):
        def transform_img(img):
            if self.transform is not None:
                img = transform(img.numpy())
            return img

        a, p, n = self.data[t[0]], self.data[t[1]], self.data[t[2]]

        img_a = transform_img(a).astype(float)
        img_p = transform_img(p).astype(float)
        img_n = transform_img(n).astype(float)

        img_a = np.expand_dims(img_a, -1)
        img_p = np.expand_dims(img_p, -1)
        img_n = np.expand_dims(img_n, -1)
        if self.out_triplets:
            return img_a, img_p, img_n
        else:
            return img_a, img_p

    def __len__(self):
        '''Denotes the number of batches per epoch'''
        return int(np.floor(len(self.triplets) / self.batch_size))
                
    def __getitem__(self, index):
        y = np.zeros((self.batch_size, 1))
        img_a = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        img_p = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        if self.out_triplets:
            img_n = np.empty((self.batch_size,) + self.dim + (self.n_channels,))
        for i in range(self.batch_size):
            t = self.triplets[self.batch_size*index + i]    
            img_a_t, img_p_t, img_n_t = self.get_image(t)
            img_a[i] = img_a_t
            img_p[i] = img_p_t
            if self.out_triplets:
                img_n[i] = img_n_t

        if self.dog:
            img_a = dogs(img_a)
            img_p = dogs(img_p)
            img_n = dogs(img_n)

        return {'a': img_a, 'p': img_p, 'n': img_n}, y

    def on_epoch_end(self):
        # 'Updates indexes after each epoch'
        self.triplets = generate_triplets(self.labels, self.num_triplets, 32)


class DenoiseHPatchesImproved(DenoiseHPatches):
    """Class for loading an HPatches sequence from a sequence folder.
       Data can be presupplied for this class.
       """
    itr = tps
    def __init__(self, seqs, batch_size = 32, dump=None, nodisk=False, suff=''):
        ''' If you already have the NPatches data in a folder, then just
            supply it as a 3-tuple in presupplied_data. This speeds up
            initialization.
        '''
        self.batch_size = batch_size
        self.dim = (32, 32)
        self.n_channels = 1
        (self.all_paths, self.sequences, self.sequences_n) = get_denoiser_data(seqs, self.dim, self.n_channels, dump=dump, nodisk=nodisk, suff=suff)
        self.on_epoch_end()


def get_denoiser_data(all_hpatches_dirs, dim=(32,32), num_channels=1, dump=None, nodisk=False, suff=''):
    ''' This will generate and/or load the data needed for the Denoise Generator.
        Set dump to the place you want to store the files.
        Set nodisk to true to rewrite the files.
        suff could be 'train' OR 'test' to generate a different generator for each of
        train and test.'''
    if dump and not nodisk and os.path.exists(os.path.join(dump, "./denoise_generator_{}paths.dat".format(suff))):
        print('Opening existing data')
        with open(os.path.join(dump, "./denoise_generator_{}paths.dat".format(suff)), 'rb') as denoise_file:
            all_paths = pickle.load(denoise_file)
        with open(os.path.join(dump, "./denoise_generator_{}seq.dat".format(suff)), 'rb') as denoise_file:
            sequences = pickle.load(denoise_file)
        with open(os.path.join(dump, "./denoise_generator_{}seqn.dat".format(suff)), 'rb') as denoise_file:
            sequences_n = pickle.load(denoise_file)
    else:
        if not nodisk and not dump:
            raise ValueError('Please supply the dump path or set nodisk to True to avoid using the disk.')
        print('Generating new data')
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
        if dump and not nodisk:
            print('Dumping generated data')
            with open(os.path.join(dump, "./denoise_generator_{}paths.dat".format(suff)), 'w+b') as denoise_file:
                pickle.dump(all_paths, denoise_file)
            with open(os.path.join(dump, "./denoise_generator_{}seq.dat".format(suff)), 'w+b') as denoise_file:
                pickle.dump(sequences, denoise_file)
            with open(os.path.join(dump, "./denoise_generator_{}seqn.dat".format(suff)), 'w+b') as denoise_file:
                pickle.dump(sequences_n, denoise_file)
    print('Finished')
    return (all_paths,  sequences, sequences_n)