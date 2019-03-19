''' DnCNN model architecture - credit: https://github.com/cszn/DnCNN/blob/master/TrainingCodes/dncnn_keras/main_train.py. '''

import argparse
import re
import os, glob, datetime
import numpy as np
import keras
import keras.backend as K

from keras.layers import  Input,Conv2D,BatchNormalization,Activation,Subtract
from keras.models import Model
from keras.optimizers import Adam

from dl2019.models.callback import DnCNNLRScheduler

def sum_squared_error(y_true, y_pred):
    #return K.mean(K.square(y_pred - y_true), axis=-1)
    #return K.sum(K.square(y_pred - y_true), axis=-1)/2
    return K.sum(K.square(y_pred - y_true))/2

class DnCNN(Model):
    ''' DnCNN Network: https://arxiv.org/pdf/1608.03981.pdf. '''
    def __init__(self, shape, depth=17, filters=64, image_channels=1, use_bnorm=True):
        layer_count = 0
        inpt = Input(shape=shape,name = 'input'+str(layer_count))
        # 1st layer, Conv+relu
        layer_count += 1
        x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',name = 'conv'+str(layer_count))(inpt)
        layer_count += 1
        x = Activation('relu',name = 'relu'+str(layer_count))(x)
        # depth-2 layers, Conv+BN+relu
        for i in range(depth-2):
            layer_count += 1
            x = Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1),kernel_initializer='Orthogonal', padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
            if use_bnorm:
                layer_count += 1
                #x = BatchNormalization(axis=3, momentum=0.1,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            x = BatchNormalization(axis=3, momentum=0.0,epsilon=0.0001, name = 'bn'+str(layer_count))(x)
            layer_count += 1
            x = Activation('relu',name = 'relu'+str(layer_count))(x)
        # last layer, Conv
        layer_count += 1
        x = Conv2D(filters=image_channels, kernel_size=(3,3), strides=(1,1), kernel_initializer='Orthogonal',padding='same',use_bias = False,name = 'conv'+str(layer_count))(x)
        layer_count += 1
        x = Subtract(name = 'subtract' + str(layer_count))([inpt, x])   # input - noise
        super(DnCNN, self).__init__(inputs = inpt, outputs=x)

    def compile(self, loss=sum_squared_error, optimizer=None, metrics=['mae']):
        if not optimizer:
            optimizer = Adam(lr=1e-3)
        super(DnCNN, self).compile(loss=loss, optimizer=optimizer, metrics=metrics)

    def fit_generator(self, generator, callbacks=None, **kwargs):
        # Add the learning rate changer callback
        initial_learning_rate = K.eval(self.optimizer.lr)
        if not callbacks:
            callbacks = []
        callbacks.append(DnCNNLRScheduler(initial_learning_rate))
        # Super (default function)
        super(DnCNN, self).fit_generator( generator, callbacks=callbacks, **kwargs)
