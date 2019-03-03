from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input, Lambda, Reshape
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization 
from keras.layers import Input, UpSampling2D, concatenate

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

    def compile(self, l=None, o=None, m=['mae']):
        super(BaselineDenoise, self).compile(loss=l, optimizer=o, metrics=m)


class BaselineDescriptor(Sequential):
    def __init__(self, shape):
          init_weights = keras.initializers.he_normal()
  
          self.add(Conv2D(32, 3, padding='same', input_shape=shape, use_bias = True, kernel_initializer=init_weights))
          self.add(BatchNormalization(axis = -1))
          self.add(Activation('relu'))

          self.add(Conv2D(32, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
          self.add(BatchNormalization(axis = -1))
          self.add(Activation('relu'))

          self.add(Conv2D(64, 3, padding='same', strides=2, use_bias = True, kernel_initializer=init_weights))
          self.add(BatchNormalization(axis = -1))
          self.add(Activation('relu'))

          self.add(Conv2D(64, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
          self.add(BatchNormalization(axis = -1))
          self.add(Activation('relu'))

          self.add(Conv2D(128, 3, padding='same', strides=2,  use_bias = True, kernel_initializer=init_weights))
          self.add(BatchNormalization(axis = -1))
          self.add(Activation('relu'))

          self.add(Conv2D(128, 3, padding='same', use_bias = True, kernel_initializer=init_weights))
          self.add(BatchNormalization(axis = -1))
          self.add(Activation('relu'))
          self.add(Dropout(0.3))

          self.add(Conv2D(128, 8, padding='valid', use_bias = True, kernel_initializer=init_weights))
          
          # Final descriptor reshape
          descriptor_model.add(Reshape((128,)))


def triplet_loss(x):
  #output_dim = 128
  a, p, n = x
  _alpha = 1.0
  positive_distance = K.mean(K.square(a - p), axis=-1)
  negative_distance = K.mean(K.square(a - n), axis=-1)
  
  return K.expand_dims(K.maximum(0.0, positive_distance - negative_distance + _alpha), axis = 1)
