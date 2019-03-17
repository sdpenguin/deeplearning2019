import numpy as np
import os

from keras.callbacks import Callback

class SaveProgress(Callback):
    def __init__(self, dump='.', curr_epoch=0):
        ''' Specify a current epoch if one exists. '''
        self.dump = dump
        self.curr_epoch = curr_epoch
        super(SaveProgress, self).__init__()


    def on_epoch_end(self, epoch, logs=None):
        ''' Saves model losses and weights'''
        self.curr_epoch = self.curr_epoch + 1
        #val_loss = logs['val_loss']
        #train_loss = logs['loss']
        metrics = [] # This is needed if different models use different loss functions
        for k in self.params['metrics']:
            if k in logs:
                metrics.append((k, logs[k]))
        # Save current losses
        try:
            with open(os.path.join(self.dump, str(self.curr_epoch) + '.npy'), 'w+b') as losses_file:
                np.save(losses_file, np.array(metrics))
        except TypeError:
            with open(os.path.join(self.dump, str(self.curr_epoch) + '.npy'), 'w+') as losses_file:
                np.save(losses_file, np.array(metrics))
        # Save the current weights
        self.model.save_weights(os.path.join(self.dump, str(self.curr_epoch) + '.h5'))
