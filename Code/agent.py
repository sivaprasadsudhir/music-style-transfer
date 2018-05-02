from keras.callbacks import ModelCheckpoint, EarlyStopping

from autoencoder import Autoencoder
from spectrogram import Spectrogram

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from keras.utils import print_summary
from keras.models import load_model

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
			pass
			# self.network = Autoencoder(load_model = True,
			# 		                         load_path = self.trained_weights_path)
		else:
			self.get_data()
			self.network = Autoencoder(input_shape=self.x_train.shape[1:],                             learning_r=params['learning_rate'])
			# pdb.set_trace()
			self.network.save_model(self.trained_weights_path + 'model.h5',
									self.trained_weights_path + 'encoder.h5',
									self.trained_weights_path + 'decoder.h5')
		# self.load_network()

	# this is har coded for keyboard to mallet
	def get_data(self):

		print ('Reading Data...')
		f = open('../Data/keyboard_mallet_shared.json')
		file_list = json.load(f)
		f.close()

		file_list = file_list[:100]
		file_list = [fname.replace("keyboard", "mallet", 2) for fname in file_list]
		print(file_list[:5])

		file_train, file_test = train_test_split(file_list, test_size=0.3, random_state=0)

		spec_obj = Spectrogram(filenames = file_train)
		self.x_train = spec_obj.spectrogram

		spec_obj.wav_to_spectrogram(file_test)
		self.x_test = spec_obj.spectrogram

	def train(self):

		print ('Training Autoencoder...')

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
				'model_weights.h5', self.trained_weights_path + \
				'encoder_weights.h5')

	def test(self):
		keyboard_encoder = load_model("../Checkpoints/keyboard/encoder.h5")
		keyboard_encoder.load_weights("../Checkpoints/keyboard/encoder_weights.h5")
		# print(keyboard_encoder.layers[-1].get_weights())
		mallet_encoder = load_model("../Checkpoints/mallet/encoder.h5")
		mallet_encoder.load_weights("../Checkpoints/mallet/encoder_weights.h5")
		# print(mallet_encoder.layers[-1].get_weights())
		keyboard_embeddings = []
		mallet_embeddings = []

		print ('Reading Data...')
		f = open('../Data/keyboard_mallet_shared.json')
		file_list = json.load(f)
		f.close()

		file_list = file_list[:200]

		for filename in file_list:
			test_data = Spectrogram(filenames=[filename])
			keyboard_embeddings.append(keyboard_encoder.predict(test_data.spectrogram))
			filename = filename.replace("keyboard", "mallet", 2)
			test_data = Spectrogram(filenames=[filename])
			mallet_embeddings.append(mallet_encoder.predict(test_data.spectrogram))
		keyboard_embeddings = np.squeeze(np.array(keyboard_embeddings))
		mallet_embeddings = np.squeeze(np.array(mallet_embeddings))

		lr = LinearRegression()
		lr.fit(keyboard_embeddings, mallet_embeddings)
		print(lr.score(keyboard_embeddings, mallet_embeddings))



	def load_network(self):
		if eval(self.params['load_weights']):
			model_load_file = self.trained_weights_path + 'model_weights'
			encoder_load_file = self.trained_weights_path + 'encoder_weights'
			load_ckpt_number = self.params['load_ckpt_number']
			if load_ckpt_number != 0:
				model_load_file += '-0-' + str(load_ckpt_number)
			model_load_file += '.h5'
			encoder_load_file += '.h5'

			self.network.load_model_weights(model_load_file,
			                                encoder_load_file)
