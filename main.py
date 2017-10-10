"""
[Oct 6, 2017]
Changing cifar10 to mnist. 
Moreover, I will reduce the number of data to reduce the computational complexity. 
"""
from keras.preprocessing import image
from keras.datasets import cifar10
from keras.datasets import mnist
from keras.optimizers import Adam
import numpy as np
import models
import train
import utils

# import os
# os.environ["CUDA_VISIBLE_DEVICES"]="1"

# =============================================================
# Load data
load_data_name = 'cifar10'  # or 'mnist'
# load_data_name = 'mnist'


def select_class(X_train, y_train, class_id=0):
    X_train = X_train[y_train == class_id]
    y_train = y_train[y_train == class_id]
    return X_train, y_train


def load_selected_data():
    print('Stage-1. Load data and class selection.')

    (X_train, y_train), (X_test, y_test) = mnist.load_data()
    print("Before selection:", X_train.shape)
    X_train, y_train = select_class(X_train, y_train, class_id=0)
    X_test, y_test = select_class(X_test, y_test, class_id=0)
    print("After selection:", X_train.shape)

    X_train = X_train.reshape(list(X_train.shape) + [1])
    X_test = X_test.reshape(list(X_test.shape) + [1])
    print("After reshape:", X_train.shape)

    return (X_train, y_train), (X_test, y_test)

if load_data_name == 'mnist':
    (X_train, y_train), (X_test, y_test) = load_selected_data()
elif load_data_name == 'cifar10':
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
else:
    raise ValueError(
        'load_data_name of {} is not supported!'.format(load_data_name))
print('Shape of input data:', X_train.shape)

# image parameters
img_size = X_train.shape[1]  # Size of square image
channels = X_train.shape[3]  # 1 for grayscale, 3 for color
print('img_size, channels:', img_size, channels)
# =============================================================

# Training parameters
epochs = 1000
batches_per_epoch = 150
batch_size = 16  # 16 is batch_size
gamma = .5  # between 0 and 1

# Model parameters
z = 100  # Generator input
h = 128  # Autoencoder hidden representation
adam = Adam(lr=0.00005)  # lr: between 0.0001 and 0.00005
# In the paper, Adam's learning rate decays if M stalls. This is not
# implemented.

# Build models
generator = models.decoder(z, img_size, channels)
discriminator = models.autoencoder(h, img_size, channels)
gan = models.gan(generator, discriminator)

generator.compile(loss=models.l1Loss, optimizer=adam)
discriminator.compile(loss=models.l1Loss, optimizer=adam)
gan.compile(loss=models.l1Loss, optimizer=adam)

dataGenerator = image.ImageDataGenerator(preprocessing_function=utils.dataRescale)
batchIterator = dataGenerator.flow(X_train, batch_size=batch_size)

trainer = train.GANTrainer(generator, discriminator,
                           gan, batchIterator, saveSampleSwatch=True)
trainer.train(epochs, batches_per_epoch, batch_size, gamma)
