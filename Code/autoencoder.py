from keras.layers import Input, Dense
from keras.models import Model
from keras.optimizers import Adam
from keras.models import load_model
from keras.layers import Conv2D, Conv1D, Conv2DTranspose, 
from keras.layers import MaxPooling2D, UpSampling2D, Reshape, Flatten
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
		self.input = Input(shape=input_shape)
		print input_shape

		# "encoded" is the encoded representation of the input
		encoded = Conv2D(filters=40, kernel_size=(3, 3), strides=(3, 3),
		                 activation='relu', padding='valid')(self.input)
		encoded = Conv2D(filters=80, kernel_size=(2, 2),
		                 activation='relu', padding='valid')(encoded)
		encoded = Conv2D(filters=120, kernel_size=(2, 2), strides=(2, 2),
		                 activation='relu', padding='valid')(encoded)
		encoded = Conv2D(filters=200, kernel_size=(21, 47),
		                 activation='relu', padding='valid')(encoded)
		self.encoded=encoded

		decoded = Conv2DTranspose(filters=200, kernel_size=(21, 47),
                    activation='relu', padding='valid')(self.encoded)
		decoded = Conv2DTranspose(filters=120, kernel_size=(2, 2),
					strides=(2, 2), activation='relu', padding='valid')(decoded)
		decoded = Conv2DTranspose(filters=80, kernel_size=(2, 2),
					activation='relu', padding='valid')(decoded)
		decoded = Conv2DTranspose(filters=40, kernel_size=(3, 3),
					strides=(3, 3), activation='relu', padding='valid')(decoded)
		decoded = Conv2D(filters=1, kernel_size=(1, 1), activation='relu',
					padding='valid')(decoded)
		self.decoded = decoded

		# this model maps an input to its reconstruction
		self.model = Model(self.input, self.decoded)

		self.model.compile(optimizer=Adam(lr=learning_r), loss='mse')
		print_summary(self.model, line_length=80)

	def save_model_weights(self, model_path):
		self.model.save_weights(model_path)

	def load_model_weights(self, model_path):
		self.model.load_weights(model_path)

	def save_model(self, model_path, encoder_path, decoder_path):
		self.model.save(model_path)

	def load_model(self, model_path, encoder_path, decoder_path):
		self.model = load_model(model_path)
