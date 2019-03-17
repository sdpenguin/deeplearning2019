import keras

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.layers import Input, UpSampling2D, concatenate
from keras.initializers import he_normal

class UNetDenoise(Model):
    def __init__(self, shape):
        # Input
        inputs = Input(shape)
        # Conv Layer 1 (Depth 0)
        conv1 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        conv2 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv2)
        # Conv Layer 2 (Depth 1)
        conv3 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        conv4 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
        pool2 = MaxPooling2D(pool_size=(2, 2))(conv4)
        # Conv Layer 3 (Depth 2)
        conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
        conv6 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
        pool3 = MaxPooling2D(pool_size=(2, 2))(conv6)
        # Conv Layer 4 (Depth 3)
        conv7 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
        conv8 = Conv2D(512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv7)
        pool4 = MaxPooling2D(pool_size=(2, 2))(conv8)
        # Conv Layer 5 (Depth 4)
        conv9 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
        conv10 = Conv2D(1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
        upsample1 = UpSampling2D(size = (2,2))(conv10)
        # Conv Layer 6 (Depth 3)
        concat1 = concatenate([conv8,upsample1], axis = -1)
        conv11 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat1)
        conv12 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv11)
        upsample2 = UpSampling2D(size = (2,2))(conv12)
        # Conv Layer 7 (Depth 2)
        concat2 = concatenate([conv6, upsample2], axis = -1)
        conv13 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat2)
        conv14 = Conv2D(256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv13)
        upsample3 = UpSampling2D(size = (2,2))(conv14)
        # Conv Layer 8 (Depth 1)
        concat3 = concatenate([conv4, upsample3], axis = -1)
        conv15 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat3)
        conv16 = Conv2D(128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv15)
        upsample4 = UpSampling2D(size = (2,2))(conv16)
        # Conv Layer 9 (Depth 0)
        concat4 = concatenate([conv2, upsample4], axis = -1)
        conv17 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(concat4)
        conv18 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv17)

        #Output
        output = Conv2D(1, 1, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv18)

        super(UNetDenoise, self).__init__(inputs = inputs, outputs = output)

    def compile(self, loss='mean_absolute_error', optimizer=None, metrics=['mae']):
        if not optimizer:
            optimizer = keras.optimizers.sgd(lr=1e-5, momentum=0.9, nesterov=True)
        super(UNetDenoise, self).compile(loss=loss, optimizer=optimizer, metrics=metrics)
