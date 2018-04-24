from keras.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import Autoencoder
from spectrogram import Spectrogram

from sklearn.model_selection import train_test_split
from keras.utils import print_summary

import numpy as np
import argparse
import json
import sys, os, errno
import pdb
import copy

class Agent(object):

	def __init__(self, params):
		
		self.params = params
		self.trained_weights_path = params['trained_weights_path']

		if eval(self.params['load_model']):
			self.network = Autoencoder(load_model = True, 
					load_path = self.params['trained_weights_path'])
		else:
			self.get_data()
			self.network = Autoencoder(input_shape = self.x_train.shape[1:], 
												learning_r = params['learning_rate'])
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')
		self.load_network()

	def get_data(self):

		print ('Reading Data...')

		# file_list = os.listdir(self.params['data_path'])
		# file_list = [os.path.join(self.params['data_path'], i)
		# 		for i in file_list[5:15]]

		file_list = [self.params['data_path'] + 'mallet_acoustic_000-091-075.wav']		

		# print(file_list)

		x_data = Spectrogram(filenames = file_list)

		self.x_train = self.x_test = x_data.spectrogram
		# self.x_train, self.x_test = train_test_split(x_data.spectrogram,
		# 												test_size=0.3)

	def train(self):

		print ('Training...')

		filename = self.trained_weights_path + 'model_weights-'+ str(self.params['load_ckpt_number']) + '-{epoch:02d}.h5'

		checkpoint = ModelCheckpoint(filename,
									monitor='val_acc', verbose=1,
									mode='max', save_weights_only=True,
									period=self.params['save_epoch_period'])

		# callbacks_list = [EarlyStopping(patience=6), checkpoint]

		self.network.model.fit(self.x_train, self.x_train,
		   			           epochs = self.params['num_epochs'],
							   batch_size = 256,
							   shuffle = True,
							   validation_data = (self.x_test, self.x_test),
							   )

		self.network.save_model_weights(self.trained_weights_path + 'model_weights.h5')

	def test(self, filenames):
		# filenames = filenames[5:15]
		for filename in filenames:
			test_data = Spectrogram(filenames=[filename])
			decoded_spectrogram = self.network.model.predict(test_data.spectrogram)
			#print_summary(self.network.model, line_length=80)

			# test_data.spectrogram_to_wav(filename=filename,
			# 							 spectrogram=copy.deepcopy(decoded_spectrogram))

			test_data.visualize(filename=filename,
			                    spectrogram = decoded_spectrogram)

	def load_network(self):
		if eval(self.params['load_weights']):
			model_load_file = self.trained_weights_path + 'model_weights'
			load_ckpt_number = self.params['load_ckpt_number']
			if load_ckpt_number != 0:
				model_load_file += '-' + str(load_ckpt_number)
			model_load_file += '.h5'

			self.network.load_model_weights(model_load_file)