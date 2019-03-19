import numpy as np
import os

from keras.callbacks import Callback, LearningRateScheduler

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

class DnCNNLRScheduler(LearningRateScheduler):
    def __init__(self, initial_lr):
        super(DnCNNLRScheduler, self).__init__(self.get_lr_schedule(initial_lr))

    def get_lr_schedule(self, initial_lr):
        ''' Decorator to get the initial_lr. '''
        def lr_schedule(epoch):
            if epoch<=30:
                lr = initial_lr
            elif epoch<=60:
                lr = initial_lr/10
            elif epoch<=80:
                lr = initial_lr/20
            else:
                lr = initial_lr/20
            print('Current learning rate is %2.8f' %lr)
            return lr
        return lr_schedule