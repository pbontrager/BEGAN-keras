import numpy as np
import keras.backend as K
from keras.models import Model
from keras.layers import Input, Flatten, Dense, Reshape
# from keras.layers.core import Flatten, Dense, Reshape
# from keras.layers.convolutional import Convolution2D, UpSampling2D
from keras.layers import Conv2D, UpSampling2D

def l1Loss(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))

def shape(depth, row, col):
    if(K.image_dim_ordering() == 'th'):
        return (depth, row, col)
    else:
        return (row, col, depth)

def decoder(h, img_dim, channels, n = 128):
    '''
    The decoder model is used as both half of the discriminator and as the generator.

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    # init_dim = 8 #Starting size from the paper
    init_dim = img_dim // 2 // 2 
    layers = int(np.round(np.log2(img_dim)) - 3) 
    print("Decoder layers:", layers)
    
    mod_input = Input(shape=(h,))
    x = Dense(n*init_dim**2)(mod_input)
    x = Reshape(shape(n, init_dim, init_dim))(x)
    
    x = Conv2D(n, (3, 3), activation='elu', padding="same")(x)
    x = Conv2D(n, (3, 3), activation='elu', padding="same")(x)
    
    for i in range(layers):
        x = UpSampling2D(size=(2,2))(x)
        x = Conv2D(n, (3, 3), activation='elu', padding="same")(x)
        x = Conv2D(n, (3, 3), activation='elu', padding="same")(x)
        
    x = Conv2D(channels, (3, 3), activation='elu', padding="same")(x)
    
    return Model(mod_input,x)

def encoder(h, img_dim, channels, n = 128):
    '''
    The encoder model is the inverse of the decoder used in the autoencoder.

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    init_dim = 8
    layers = int(np.round(np.log2(img_dim)) - 2)
    print("Encoder layers:", layers)
    
    mod_input = Input(shape=shape(channels, img_dim, img_dim))
    x = Conv2D(channels, (3, 3), activation='elu', padding="same")(mod_input)
    
    for i in range(1, layers):
        x = Conv2D(i*n, (3, 3), activation='elu', padding="same")(x)
        x = Conv2D(i*n, (3, 3), activation='elu', padding="same", strides=(2,2))(x)
    
    x = Conv2D(layers*n, (3, 3), activation='elu', padding="same")(x)
    x = Conv2D(layers*n, (3, 3), activation='elu', padding="same")(x)
    
    # x = Reshape((layers*n*init_dim**2,))(x)
    x = Flatten()(x)
    x = Dense(h)(x)
    
    return Model(mod_input,x)

def autoencoder(h, img_dim, channels, n = 128):
    '''
    The autoencoder is used as the discriminator

    Keyword Arguments:
    h -- Integer size of the 1 dimensional input vector
    img_dim -- Integer size of the square output image
    channels -- 1 or 3 depending on whether the images have color channels or not.
    n -- Number of convolution filters, paper value is 128
    '''
    mod_input = Input(shape=shape(channels, img_dim, img_dim))
    x = encoder(h, img_dim, channels, n)(mod_input)
    x = decoder(h, img_dim, channels, n)(x)
    
    return Model(mod_input, x)

def gan(generator, discriminator):
    '''
    Combined generator and discriminator

    Keyword arguments:
    generator -- The instantiated generator model
    discriminator -- The instantiated discriminator model
    '''
    mod_input = generator.input
    x = generator(mod_input)
    x = discriminator(x)

    return Model(mod_input, x)