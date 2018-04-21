from keras.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import Autoencoder, SharedAutoencoder
from spectrogram import Spectrogram

from sklearn.model_selection import train_test_split
from keras.utils import print_summary

import numpy as np
import argparse
import json
import sys, os, errno
import pdb

class Agent(object):

	def __init__(self, params):

		self.params = params
		self.trained_weights_path = params['trained_weights_path']

		if not eval(self.params['train']):
			self.network = Autoencoder(load_model = True,
					                   load_path = self.trained_weights_path)
		else:
			self.get_data()
			self.network = Autoencoder(input_shape=self.x_train.shape[1:],
			                           learning_r=params['learning_rate'])
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')

	def get_data(self):

		print ('Reading Data...')

		file_list = os.listdir(self.params['data_path'])
		file_list = [os.path.join(self.params['data_path'], i)
				for i in file_list[5:15]]

		print(file_list)

		x_data = Spectrogram(filenames = file_list)

		self.x_train, self.x_test = train_test_split(x_data.spectrogram,
														test_size=0.3)

	def train(self):

		print ('Training...')

		callbacks_list = [EarlyStopping()]

		self.network.model.fit(self.x_train, self.x_train,
		   			           epochs = self.params['num_epochs'],
							   batch_size = 256,
							   shuffle = True,
							   validation_data = (self.x_test, self.x_test),
							   callbacks=callbacks_list)

		self.network.save_model_weights(self.trained_weights_path + 'model_weights.h5')

	def test(self, filenames):
		filenames = filenames[5:15]
		for filename in filenames:
			test_data = Spectrogram(filenames=[filename])
			decoded_spectrogram = self.network.model.predict(test_data.spectrogram)
			#print_summary(self.network.model, line_length=80)

			test_data.visualize(filename=filename,
			                    spectrogram = decoded_spectrogram)

class SharedAgent(Agent):

	def __init__(self, params):

		self.params = params
		self.trained_weights_path = params['trained_weights_path']

		if eval(self.params['load_model']):
			self.network = SharedAutoencoder(load_model = True,
					                   load_path = self.trained_weights_path)
		else:
			self.get_data()
			self.network = SharedAutoencoder(input_shape=self.x_train.shape[1:], learning_r=params['learning_rate'])
			# pdb.set_trace()
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')

	# this is har coded for keyboard to mallet
	def get_data(self):

		print ('Reading Data...')

		file_list = os.listdir(self.params['data_path'])
		file_list = [os.path.join(self.params['data_path'], i)
				for i in file_list[50:60]]

		file_train, file_test = train_test_split(file_list, test_size=0.3)

		spec_obj = Spectrogram(filenames = file_train)

		self.x_train = spec_obj.spectrogram
		file_train_instr2 = [filename.replace("keyboard_", "mallet_", 2) for filename in file_train]
		spec_obj.wav_to_spectrogram(file_train_instr2)
		self.y_train = np.concatenate([self.x_train, spec_obj.spectrogram], axis = -1)

		spec_obj.wav_to_spectrogram(file_test)
		self.x_test = spec_obj.spectrogram
		file_test_instr2 = [filename.replace("keyboard_", "mallet_", 2) for filename in file_test]
		spec_obj.wav_to_spectrogram(file_test_instr2)
		self.y_test = np.concatenate([self.x_test, spec_obj.spectrogram], axis = -1)


	def train(self):

		print ('Training SharedAutoencoder...')

		raw_input()

		callbacks_list = [EarlyStopping()]

		self.network.model.fit(self.x_train, self.y_train,
		   			           epochs = self.params['num_epochs'],
							   batch_size = 256,
							   shuffle = True,
							   validation_data = (self.x_test, self.y_test),
							   callbacks=callbacks_list)

		self.network.save_model_weights(self.trained_weights_path + 'model_weights.h5')

	def test(self, filenames):
		for filename in filenames:
			test_data = Spectrogram(filenames=[filename])
			decoded_spectrogram = self.network.model.predict(test_data.spectrogram)

			keyboard_out, mallet_out = np.split(decoded_spectrogram[0], 2)
			# pdb.set_trace()
			#print_summary(self.network.model, line_length=80)

			test_data.visualize(filename=filename,
			                    spectrogram = keyboard_out)
			test_data.visualize(filename=filename,
			                    spectrogram = mallet_out)