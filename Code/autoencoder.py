from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Reshape, Flatten, concatenate
from keras.utils import print_summary

import numpy as np
import pdb


class Autoencoder(object):

	def __init__(self, load_model=False, input_shape = None,
	             learning_r = None, load_path = None):
		# load or create from scratch
		if load_model:
			self.load_model(load_path + 'model.h5',
							load_path + 'encoder.h5',
							load_path + 'decoder.h5')
		else:
			self.define_entire_network(input_shape, learning_r)

	def define_entire_network(self, input_shape, learning_r):
		# this is our input placeholder
		self.input = Input(shape=input_shape)
		print input_shape

		self.encoding_dim = 16

		# "encoded" is the encoded representation of the input
		encoded = Dense(256, activation='relu')(self.input)
		encoded = Dense(128, activation='relu')(encoded)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(32, activation='relu')(encoded)
		self.encoded = Dense(self.encoding_dim, activation='relu')(encoded)

		# "decoded" is the lossy reconstruction of the input for instrument 1
		decoded = Dense(32, activation='relu')(self.encoded)
		decoded = Dense(64, activation='relu')(decoded)
		decoded = Dense(128, activation='relu')(decoded)
		decoded = Dense(256, activation='relu')(decoded)
		self.decoded = Dense(input_shape[0], activation='relu')(decoded)

		# this model maps an input to its reconstruction
		self.model = Model(self.input, self.decoded)

		self.define_encoder_network()
		self.define_decoder_network()

		self.model.compile(optimizer=Adam(lr=learning_r), loss='mse')
		print_summary(self.model, line_length=80)

	def define_encoder_network(self):
		# this model maps an input to its encoded representation
		self.encoder = Model(self.input, self.encoded)
	
	def define_decoder_network(self):
		# create a placeholder for an encoded (32-dimensional) input
		encoded_input = Input(shape=(self.encoding_dim,))

		# retrieve the last layer of the autoencoder model
		decoded_output = self.model.layers[-5](encoded_input)
		decoded_output = self.model.layers[-4](decoded_output)
		decoded_output = self.model.layers[-3](decoded_output)
		decoded_output = self.model.layers[-2](decoded_output)
		decoded_output = self.model.layers[-1](decoded_output)
		
		# create the decoder model
		self.decoder = Model(encoded_input, decoded_output)

	def save_model_weights(self, model_path, encoder_path, decoder_path):
		self.model.save_weights(model_path)
		self.encoder.save_weights(encoder_path)
		self.decoder.save_weights(decoder_path)

	def load_model_weights(self, model_path, encoder_path, decoder_path):
		self.model.load_weights(model_path)
		self.encoder.load_weights(encoder_path)
		self.decoder.load_weights(decoder_path)

	def save_model(self, model_path, encoder_path, decoder_path):
		self.model.save(model_path)
		self.encoder.save(encoder_path)
		self.decoder.save(decoder_path)

	def load_model(self, model_path, encoder_path, decoder_path):
		self.model = load_model(model_path)
		self.encoder = load_model(encoder_path)
		self.decoder = load_model(decoder_path)

