from keras.preprocessing import image
from keras.datasets import cifar10
from keras.optimizers import Adam
import numpy as np
import models
import train
import utils

#Training parameters
epochs = 100
batches_per_epoch = 150
batch_size = 16
gamma = .5 #between 0 and 1

#image parameters
img_size = 32 #Size of square image
channels = 3 #1 for grayscale

#Model parameters
z = 100 #Generator input
h = 128 #Autoencoder hidden representation
adam = Adam(lr=0.00005) #lr: between 0.0001 and 0.00005
#In the paper, Adam's learning rate decays if M stalls. This is not implemented.

#Build models
generator = models.decoder(z, img_size, channels)
discriminator = models.autoencoder(h, img_size, channels)
gan = models.gan(generator, discriminator)

generator.compile(loss=models.l1Loss, optimizer=adam)
discriminator.compile(loss=models.l1Loss, optimizer=adam)
gan.compile(loss=models.l1Loss, optimizer=adam)

#Load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
dataGenerator = image.ImageDataGenerator(preprocessing_function = utils.dataRescale)
batchIterator = dataGenerator.flow(X_train, batch_size = batch_size)

trainer = train.GANTrainer(generator, discriminator, gan, batchIterator)
trainer.train(epochs, batches_per_epoch, batch_size, gamma)