from keras.callbacks import ModelCheckpoint

from autoencoder import Autoencoder
from spectrogram import Spectrogram

from sklearn.model_selection import train_test_split

import numpy as np
import argparse
import json
import sys, os, errno
import pdb

class Agent(object):

	def __init__(self, params):
		
		self.params = params
		self.trained_weights_path = params['trained_weights_path']

		if eval(self.params['load_model']):
			self.network = Autoencoder(load_model = True, 
					load_path = self.params['trained_weights_path'])
		else:
			self.get_data()
			self.network = Autoencoder(self.x_train.shape[1:], 
												params['learning_rate'])
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')
		self.load_network()

	def get_data(self):

		print ('Reading Data...')

		file_list = os.listdir(self.params['data_path'])
		file_list = [os.path.join(self.params['data_path'], i) 
				for i in file_list]

		# print file_list
		# file_list = ["../nsynth-train/keyboard_acoustic/keyboard_acoustic_000-021-025.wav"]

		x_data = Spectrogram(filenames = file_list)

		train_index, test_index = train_test_split(xrange(len(x_data.keyboard)),
														test_size=0.3)

		self.x_train = x_data.keyboard[train_index]
		self.x_test = x_data.keyboard[test_index]

		self.y_train = x_data.mallet[train_index]
		self.y_test = x_data.mallet[test_index]
		
		# print len(self.x_train), len(self.x_test)

		# self.x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
		# self.x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
	
	def load_network(self):
		
		if eval(self.params['load_weights']):
			
			model_load_file = self.trained_weights_path + 'model_weights'
			load_ckpt_number = self.params['load_ckpt_number']
			if load_ckpt_number != 0:
				model_load_file += '-' + str(load_ckpt_number)
			model_load_file += '.h5'

			self.network.load_model_weights(model_load_file)

	def train(self):

		print ('Training...')

		# pdb.set_trace()

		# filename = self.trained_weights_path + 'model_weights-{}{epoch:02d}.h5'.format(100)
		filename = self.trained_weights_path + 'model_weights-'+ str(self.params['load_ckpt_number']) + '-{epoch:02d}.h5'
		checkpoint = ModelCheckpoint(filename,
									monitor='val_acc', verbose=1,
									mode='max', save_weights_only=True,
									period=self.params['save_epoch_period'])
		callbacks_list = [checkpoint]
		self.network.model.fit(self.x_train, self.y_train,
				epochs = self.params['num_epochs'],
				batch_size = 256,
				shuffle = True,
				callbacks = callbacks_list,
				validation_data = (self.x_test, self.y_test))

		self.network.save_model_weights(self.trained_weights_path + 'model_weights.h5')

	def test(self, filename):
		# print (filename)
		test_data = Spectrogram(filenames = [filename])
		decoded_spectrogram = self.network.model.predict(test_data.keyboard)

		# encoded_spectrogram = self.network.encoder.predict(test_data.spectrogram)
		# decoded_spectrogram = self.network.decoder.predict(encoded_spectrogram)
		f = filename.replace("keyboard", "mallet", 2)
		test_data.visualize(filename = f, spectrogram = decoded_spectrogram )

		# print (encoded_spectrogram.shape)
		# print (decoded_spectrogram.shape)
		# print (encoded_spectrogram)
		# print (decoded_spectrogram)
