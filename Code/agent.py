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
import copy
import os.path

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

		file_list = [self.params['data_path'] + 'keyboard_acoustic_000-091-075.wav']

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

		model_save_file = self.trained_weights_path + 'model_weights.h5'
		encoder_save_file = self.trained_weights_path + 'encoder_weights.h5'
		decoder_save_file = self.trained_weights_path + 'decoder_weights.h5'
		self.network.save_model_weights(model_save_file, encoder_save_file, decoder_save_file)

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
		self.load_network()

	# this is har coded for keyboard to mallet
	def get_data(self):

		print ('Reading Data...')

		file_list = os.listdir(self.params['data_path'])

		file_list_instr2 = [filename.replace("keyboard_", "mallet_", 2) for filename in file_list]
		mask = [i for i, filename in enumerate(file_list_instr2) if os.path.isfile(filename)]
		good_file_list_instr2 = [filename for i, filename in enumerate(file_list_instr2) if os.path.isfile(filename)]

		file_list = file_list[mask]
		file_list = [os.path.join(self.params['data_path'], i)
				for i in file_list[50:60]]

		file_train, file_test = train_test_split(file_list, test_size=0.3)

		file_train_instr2 = [filename.replace("keyboard_", "mallet_", 2) for filename in file_train]
		spec_obj = Spectrogram(filenames = file_train)

		self.x_train = spec_obj.spectrogram
		self.x_train = self.x_train[mask]
		spec_obj.wav_to_spectrogram(file_train_instr2)
		self.y_train = np.concatenate([self.x_train, spec_obj.spectrogram], axis = -1)

		spec_obj.wav_to_spectrogram(file_test)
		self.x_test = spec_obj.spectrogram
		file_test_instr2 = [filename.replace("keyboard_", "mallet_", 2) for filename in file_test]
		spec_obj.wav_to_spectrogram(file_test_instr2)
		self.y_test = np.concatenate([self.x_test, spec_obj.spectrogram], axis = -1)


	def train(self):

		print ('Training SharedAutoencoder...')

		callbacks_list = [EarlyStopping(patience=5)]

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
			test_data.visualize(filename=filename.replace("keyboard", "mallet", 2), spectrogram = mallet_out)
