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
import os.path

class Agent(object):

	def __init__(self, params):

		self.params = params
		self.trained_weights_path = params['trained_weights_path']

		if eval(self.params['load_model']):
			self.network = Autoencoder(load_model = True,
					                         load_path = self.trained_weights_path)
		else:
			self.get_data()
			self.network = Autoencoder(input_shape=self.x_train.shape[1:],
									learning_r=params['learning_rate'])
			# pdb.set_trace()
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')
		self.load_network()

	# this is har coded for keyboard to mallet
	def get_data(self):

		print ('Reading Data...')

		file_list = os.listdir(self.params['data_path'])
		file_list = file_list[:20]
		file_list = [os.path.join(self.params['data_path'], fname) for fname in file_list]

		file_train, file_test = train_test_split(file_list, test_size=0.3, random_state=0)

		spec_obj = Spectrogram(filenames = file_train)
		self.x_train = spec_obj.spectrogram

		spec_obj.wav_to_spectrogram(file_test)
		self.x_test = spec_obj.spectrogram

	def train(self):

		print ('Training ConvAutoencoder...')

		callbacks_list = [EarlyStopping(patience=20)]

		filename = self.trained_weights_path + 'model_weights-'+ \
					str(self.params['load_ckpt_number']) + '-{epoch:02d}.h5'

		# # checkpoint = ModelCheckpoint(filename,
		# 							monitor='val_acc', verbose=1,
		# 							mode='max', save_weights_only=True,
		# 							period=self.params['save_epoch_period'])

		#callbacks_list = [checkpoint]
		# callbacks_list = []

		htry_cb = self.network.model.fit(self.x_train, self.x_train,
		   			           epochs = self.params['num_epochs'],
							   batch_size = 256,
							   shuffle = True,
							   validation_data = (self.x_test, self.x_test),
							   callbacks=callbacks_list)

		# loss_history = htry_cb.history["val_acc"]
		with open(self.trained_weights_path + 'log_loss.txt', 'w') as file:
			file.write(json.dumps(htry_cb.history))

		self.network.save_model_weights(self.trained_weights_path + \
				'model_weights.h5')

	def test(self):
		for filename in self.params['test_data_path']:
			test_data = Spectrogram(filenames=[filename])
			decoded_spectrogram = self.network.model.predict(
					test_data.spectrogram)

			# pdb.set_trace()
			#print_summary(self.network.model, line_length=80)
			test_data.spectrogram_to_wav(filename=filename,
								spectrogram=copy.deepcopy(decoded_spectrogram),
								outfile=filename.split("/")[-1])
			test_data.visualize(filename=filename,
			                    spectrogram=decoded_spectrogram)

	def load_network(self):
		if eval(self.params['load_weights']):
			model_load_file = self.trained_weights_path + 'model_weights'
			load_ckpt_number = self.params['load_ckpt_number']
			if load_ckpt_number != 0:
				model_load_file += '-0-' + str(load_ckpt_number)
			model_load_file += '.h5'

			self.network.load_model_weights(model_load_file)
