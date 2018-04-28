from keras.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import SharedAutoencoder
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

class SharedAgent(object):

	def __init__(self, params):

		self.params = params
		self.trained_weights_path = params['trained_weights_path']

		if eval(self.params['load_model']):
			self.network = SharedAutoencoder(load_model = True,
					                         load_path = self.trained_weights_path)
		else:
			self.get_data()
			self.network = SharedAutoencoder(input_shape=self.x_train.shape[1:],
									learning_r=params['learning_rate'])
			# pdb.set_trace()
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')
		self.load_network()

	# this is har coded for keyboard to mallet
	def get_data(self):

		print ('Reading Data...')
		f = open('../Data/keyboard_mallet_shared.json')
		file_list = json.load(f)
		f.close()

		file_list = [filename for filename in file_list[0:20]]

		file_train, file_test = train_test_split(file_list, test_size=0.3, random_state=0)

		print(file_test)

		good_mallet_file_list = [filename.replace("keyboard_", "mallet_", 2)
									for filename in file_train]

		spec_obj = Spectrogram(filenames = file_train)
		self.x_train = spec_obj.spectrogram
		spec_obj.wav_to_spectrogram(good_mallet_file_list)
		self.y_train = np.concatenate([self.x_train, spec_obj.spectrogram],
										axis = -1)

		spec_obj.wav_to_spectrogram(file_test)
		self.x_test = spec_obj.spectrogram
		good_mallet_file_list = [filename.replace("keyboard_", "mallet_", 2)
									for filename in file_test]
		spec_obj.wav_to_spectrogram(good_mallet_file_list)
		self.y_test = np.concatenate([self.x_test, spec_obj.spectrogram],
										axis = -1)

	def train(self):

		print ('Training SharedAutoencoder...')

		callbacks_list = [EarlyStopping(patience=20)]

		filename = self.trained_weights_path + 'model_weights-'+ \
					str(self.params['load_ckpt_number']) + '-{epoch:02d}.h5'

		checkpoint = ModelCheckpoint(filename,
									monitor='val_acc', verbose=1,
									mode='max', save_weights_only=True,
									period=self.params['save_epoch_period'])

		#callbacks_list = [checkpoint]
		callbacks_list = []

		htry_cb = self.network.model.fit(self.x_train, self.y_train,
		   			           epochs = self.params['num_epochs'],
							   batch_size = 256,
							   shuffle = True,
							   validation_data = (self.x_test, self.y_test),
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

			keyboard_out, mallet_out = np.split(decoded_spectrogram[0], 2)
			# pdb.set_trace()
			#print_summary(self.network.model, line_length=80)
			test_data.spectrogram_to_wav(filename=filename,
								spectrogram=copy.deepcopy(keyboard_out),
								outfile=filename.split("/")[-1])
			test_data.visualize(filename=filename,
			                    spectrogram=keyboard_out)
			test_data.spectrogram_to_wav(
					filename=filename.replace("keyboard", "mallet", 2),
					spectrogram=copy.deepcopy(mallet_out),
					outfile=filename.replace("keyboard", "mallet", 2).split("/")[-1])
			test_data.visualize(
					filename=filename.replace("keyboard", "mallet", 2),
					spectrogram = mallet_out)

	def load_network(self):
		if eval(self.params['load_weights']):
			model_load_file = self.trained_weights_path + 'model_weights'
			load_ckpt_number = self.params['load_ckpt_number']
			if load_ckpt_number != 0:
				model_load_file += '-' + str(load_ckpt_number)
			model_load_file += '.h5'

			self.network.load_model_weights(model_load_file)
