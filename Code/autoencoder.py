from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model

import numpy as np
import pdb


class Autoencoder(object):

	def __init__(self, input_shape = None, learning_r = None, 
		load_model = False, load_path = None):
		# load or create from scratch
		if load_model:
			self.load_model(load_path + 'model.h5',
							load_path + 'encoder.h5',
							load_path + 'decoder.h5')
		else:
			self.define_entire_network(input_shape, learning_r)

	def define_entire_network(self, input_shape, learning_r):
		# this is the size of our encoded representations
		# 32 floats -> compression of factor 24.5, 
		# assuming the input is 784 floats
		self.encoding_dim = 16  

		# this is our input placeholder
		self.input = Input(shape=(input_shape,))

		# "encoded" is the encoded representation of the input
		encoded = Dense(128, activation='relu')(self.input)
		encoded = Dense(64, activation='relu')(encoded)
		encoded = Dense(32, activation='relu')(encoded)
		self.encoded = Dense(self.encoding_dim, activation='relu')(encoded)

		# "decoded" is the lossy reconstruction of the input
		decoded = Dense(32, activation='relu')(self.encoded)
		decoded = Dense(64, activation='relu')(self.encoded)
		decoded = Dense(128, activation='relu')(decoded)
		self.decoded = Dense(input_shape, activation='sigmoid')(decoded)

		# this model maps an input to its reconstruction
		self.model = Model(self.input, self.decoded)

		self.model.compile(optimizer=Adam(lr=learning_r),
								 loss='categorical_crossentropy')

		self.define_decoder_network()
		self.define_encoder_network()

	def define_encoder_network(self):
		# this model maps an input to its encoded representation
		self.encoder = Model(self.input, self.encoded)
	
	def define_decoder_network(self):
		# create a placeholder for an encoded (32-dimensional) input
		encoded_input = Input(shape=(self.encoding_dim,))
		
		# retrieve the last layer of the autoencoder model
		decoded_output = self.model.layers[-3](encoded_input)
		decoded_output = self.model.layers[-2](decoded_output)
		decoded_output = self.model.layers[-1](decoded_output)
		
		# create the decoder model
		self.decoder = Model(encoded_input, decoded_output)

	def save_model_weights(self, model_path):
		self.model.save_weights(model_path)

	def load_model_weights(self, model_path):
		self.model.load_weights(model_path)

	def save_model(self, model_path, encoder_path, decoder_path):
		self.model.save(model_path)
		self.encoder.save(encoder_path)
		self.decoder.save(decoder_path)

	def load_model(self, model_path, encoder_path, decoder_path):
		self.model = load_model(model_path)
		self.encoder = load_model(encoder_path)
		self.decoder = load_model(decoder_path)
