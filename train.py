import os
import time
import utils
import models
import numpy as np
from keras.utils import generic_utils
from keras import backend as k

class GANTrainer:
	def __init__(self, gen, disc, gan, data, kLambda = .001, logEpochOutput = True, saveModelFrequency = 5, 
                 sampleSwatch = True, saveSampleSwatch = False):
		'''
		Class contains all the default values for training a particular GAN

		Keyword Arguments:
		gen -- Generator Model to be trained
		disc -- Discriminator Model to be trained
		gan -- Combined Generator and Discriminator Model
		data -- DataGenerator that outputs real data to train with each batch

		Optional Arguments:
		kLambda -- k learning rate, value is set from the paper
		logEpochOutput -- Whether to save each output's values
		saveModelFrequency -- How many epochs to wait between saveing a model
		sampleSwatch -- Whether to output an example of the data. Top 8 represent training data, bottom 8 represent real data.
		saveSampleSwatch -- Whether to keep each swatch or overwrite it with the next one.
		'''
		self.generator = gen
		self.discriminator = disc
		self.gan = gan
		self.dataGenerator = data
		try:
			self.dataGenerator.next()
		except:
			raise Exception('Data is expected to be a DataGenerator')

		self.z = self.generator.input_shape[-1]
		self.epsilon = k.epsilon()
		self.kLambda = kLambda
		self.logEpochOutput = logEpochOutput
		self.saveModelFrequency = saveModelFrequency
		self.sampleSwatch = sampleSwatch
		self.saveSampleSwatch = saveSampleSwatch

		self.k = self.epsilon #If k = 0, like in the paper, Keras returns nan values
		self.firstEpoch = 1


	def train(self, nb_epoch, nb_batch_per_epoch, batch_size, gamma, path = ""):
		'''
		Train a Generator network and Discriminator Method using the BEGAN method. The networks are updated sequentially unlike what's done in the paper.

		Keyword Arguments:
		nb_epoch -- Number of training epochs
		batch_size -- Size of a single batch of real data.
		nb_batch_per_epoch -- Number of training batches to run each epoch.
		gamma -- Hyperparameter from BEGAN paper to regulate proportion of Generator Error over Discriminator Error. Defined from 0 to 1.
		path -- Optional parameter specifying location to save output file locations. Starts from the working directory.
		'''
		for e in range(self.firstEpoch, self.firstEpoch + nb_epoch):
			progbar = generic_utils.Progbar(nb_batch_per_epoch*batch_size)
			start = time.time()
			
			for b in range(nb_batch_per_epoch):
				zD = np.random.uniform(-1,1,(batch_size, self.z))
				zG = np.random.uniform(-1,1,(batch_size*2, self.z)) #
				            
				#Train D
				real = self.dataGenerator.next()
				d_loss_real = self.discriminator.train_on_batch(real, real)
				
				gen = self.generator.predict(zD)
				weights = -self.k*np.ones(batch_size)
				d_loss_gen = self.discriminator.train_on_batch(gen, gen, weights)
				
				d_loss = d_loss_real + d_loss_gen
				
				#Train G
				self.discriminator.trainable = False
				target = self.generator.predict(zG)
				g_loss = self.gan.train_on_batch(zG, target)
				self.discriminator.trainable = True
				
				#Update k
				self.k = self.k + self.kLambda*(gamma*d_loss_real - g_loss)
				self.k = min(max(self.k, self.epsilon), 1)
				
				#Report Results
				m_global = d_loss + np.abs(gamma*d_loss_real - g_loss)
				progbar.add(batch_size, values=[("M", m_global),("Loss_D", d_loss),("Loss_G", g_loss),("k", self.k)])
				
				if(self.logEpochOutput and b == 0):
					with open(os.getcwd() + path +'/output.txt', 'a') as f:
						f.write("{}, M: {}, Loss_D: {}, LossG: {}, k: {}\n".format(e, m_global, d_loss, g_loss, self.k))
				
				if(self.sampleSwatch and b % (nb_batch_per_epoch / 2) == 0):
					if(self.saveSampleSwatch):
						genName = '/generatorSample_{}_{}.png'.format(e, int(b/nb_batch_per_epoch/2))
						discName = '/discriminatorSample_{}_{}.png'.format(e, int(b/nb_batch_per_epoch/2))
					else:
						genName = '/currentGeneratorSample.png'
						discName = '/currentDiscriminatorSample.png'
					utils.plotGeneratedBatch(real, gen, path + genName)
					utils.plotGeneratedBatch(self.discriminator.predict(real), target, path + discName)
	
	
			print('\nEpoch {}/{}, Time: {}'.format(e + 1, nb_epoch, time.time() - start))
			
			if(e % self.saveModelFrequency == 0):
				utils.saveModelWeights(self.generator, self.discriminator, e, path)