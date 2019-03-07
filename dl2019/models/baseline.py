import keras

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from keras.layers import Input, UpSampling2D, concatenate
from keras.initializers import he_normal

class BaselineDenoise(Model):
    def __init__(self, shape):
        # Input
        inputs = Input(shape)
        # Encoder
        conv1 = Conv2D(16, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
        pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
        # Bottleneck
        conv2 = Conv2D(32, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
        # Decoder
        up3 = Conv2D(64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv2))
        merge3 = concatenate([conv1,up3], axis = -1)
        conv3 = Conv2D(64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge3)
        # Output
        conv4 = Conv2D(1, 3,  padding = 'same')(conv3)

        super(BaselineDenoise, self).__init__(inputs = inputs, outputs = conv4)

    def compile(self, loss=None, metrics=['mae']):
        optimizer = keras.optimizers.Adam(lr=1e-5, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        super(BaselineDenoise, self).compile(loss=l, optimizer=optimizer, metrics=m)


class BaselineDescriptor(Model):
    """ A triplet loss baseline descriptor model that implements a version of HardNet.
        To change the triplet loss model, simply inherit from this class and change initalise_sequential.
    """
    def __init__(self, shape):
        super(BaselineDescriptor, self).__init__()
        # Initialise a Sequential model
        self.seq_model = Sequential()
        self.initalise_sequential(shape)
        # Triplet loss inputs and outputs
        xa = Input(shape=shape, name='a')
        ya = self.seq_model(xa)
        xp = Input(shape=shape, name='p')
        yp = self.seq_model(xp)
        xn = Input(shape=shape, name='n')
        yn = self.seq_model(xn)
        inputs = [xa, xp, xn]
        outputs = [ya, yp, yn]
        # Add a Lambda layer to calculate the loss and apply to outputs
        loss = Lambda(triplet_loss)(outputs)
        # Initialise the model
        super(BaselineDescriptor, self).__init__(inputs=inputs, outputs=loss)


    def initalise_sequential(self, shape):
        #Weight Initialisation
        init_weights = he_normal() # Initialise the weights from ~N(0, 2/32)
        # Convolution and then batch normalization
        self.seq_model.add(Conv2D(32, 3, padding='same', input_shape=shape, use_bias = True, kernel_initializer=init_weights))
        self.seq_model.add(BatchNormalization(axis = -1))
        self.seq_model.add(Activation('relu'))
        # Convolution and then batch normalization
        self.seq_model.add(Conv2D(32, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
        self.seq_model.add(BatchNormalization(axis = -1))
        self.seq_model.add(Activation('relu'))
        # Convolution and then batch normalization
        self.seq_model.add(Conv2D(64, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
        self.seq_model.add(BatchNormalization(axis = -1))
        self.seq_model.add(Activation('relu'))
        # Convolution and then batch normalization
        self.seq_model.add(Conv2D(64, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
        self.seq_model.add(BatchNormalization(axis = -1))
        self.seq_model.add(Activation('relu'))
        # Convolution and then batch normalization
        self.seq_model.add(Conv2D(128, 3, padding='same', strides=2,  use_bias = True, kernel_initializer=init_weights))
        self.seq_model.add(BatchNormalization(axis = -1))
        self.seq_model.add(Activation('relu'))
        # Convolution and then batch normalization and dropout
        self.seq_model.add(Conv2D(128, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
        self.seq_model.add(BatchNormalization(axis = -1))
        self.seq_model.add(Activation('relu'))
        # Dropout at the end of this layer
        self.seq_model.add(Dropout(0.3))
        # Final convolution layer
        self.seq_model.add(Conv2D(128, 8, padding='valid', use_bias = True, kernel_initializer=init_weights))
        # Final descriptor reshape
        self.seq_model.add(Reshape((128,)))

    def compile(self, loss=None, metrics=['mae']):
        optimizer = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        super(BaselineDescriptor, self).compile(loss=loss, optimizer=optimizer, metrics=metrics)


def triplet_loss(x):
  #output_dim = 128
  a, p, n = x
  _alpha = 1.0
  positive_distance = K.mean(K.square(a - p), axis=-1)
  negative_distance = K.mean(K.square(a - n), axis=-1)
  
  return K.expand_dims(K.maximum(0.0, positive_distance - negative_distance + _alpha), axis = 1)
