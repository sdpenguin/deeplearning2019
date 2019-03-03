import os
import numpy as np
import glob

def get_latest_epoch(dir='.'):
    ''' Gets the number of the latest epoch in dir as determined by the files.'''
    directory = os.path.join(dir, '*.h5')
    files = glob.glob(directory)
    if len(files) is not 0:
        file_numbers = [int(os.path.split(os.path.splitext(x)[0])[-1]) for x in files]
        return np.max(file_numbers)
    else:
        return 0
