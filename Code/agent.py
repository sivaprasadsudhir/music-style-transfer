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

		# file_list = [filename for filename in file_list[0:20]]

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

		length_in_sec = 631

		for filename in self.params['test_data_path']:
			test_data = Spectrogram(filenames=[filename])

			# pdb.set_trace()

			a = test_data.spectrogram.reshape(1, 257, length_in_sec, 2)
			mallet_out_full = []
			x = a.shape[2]/55
			# for i in xrange(1, a.shape[2]/x):
			for i in xrange(1, 2):
				test_spectrogram = a[:, :, x*(i-1) : x*(i-1) + 251 , :]
				test_spectrogram = test_spectrogram.reshape(len(test_spectrogram), np.prod(test_spectrogram.shape[1:]))
				decoded_spectrogram = self.network.model.predict(
					test_spectrogram)

				keyboard_out, mallet_out = np.split(decoded_spectrogram[0], 2)
			
				# This code stitches the sliced mallet output consecutively
				# sliced_mallet = mallet_out.reshape(1, 257, 251, 2)
				# sliced_mallet = sliced_mallet[:, :, :x, :]
				# mallet_out_full.extend(sliced_mallet.reshape(len(sliced_mallet), np.prod(sliced_mallet.shape[1:]))[0])

				mallet_out_full.extend(keyboard_out)
				# This code adds t

				# print (i)
				# print (x*(i-1), x*(i-1) + 251)

				print (x*i + 251, a.shape[2])
				if x*i + 251 > a.shape[2]:
					break
				# print ()
				# pdb.set_trace()
				# mallet_out_full
				# pdb.set_trace()
				#print_summary(self.network.model, line_length=80)
				
				# test_data.spectrogram_to_wav(filename=filename,
				# 					spectrogram=copy.deepcopy(keyboard_out),
				# 					outfile=filename.split("/")[-1])
				
				# test_data.visualize(filename=filename,
				#                     spectrogram=keyboard_out)

			# print (i)
			# pdb.set_trace()

			mallet_out_full = np.array(mallet_out_full)

			test_data.spectrogram_to_wav(
					filename=filename,
					spectrogram=copy.deepcopy(mallet_out_full),
					)

			
			test_data.visualize(
					filename=filename.replace("keyboard", "mallet", 2),
					spectrogram = mallet_out_full)

	def test_split(self):

		x = 251
		all_mallet = []
		for filename in self.params['test_data_path']:
		# filenames = filenames[5:15]
		# for filename in filenames:
			for i in xrange(1, 2):
				# test_spectrogram = a[:, :, x*(i-1) : x*(i-1) + 251 , :]

				print(filename)
				test_data = Spectrogram(filenames=[filename])
				a = test_data.spectrogram.reshape(1, 257, 631, 2)

				first_spec = a[:, :, x*(i-1):x*i, :]
				zero_spec = np.zeros((1,257,251-x,2))

				new_spec = np.concatenate([first_spec, zero_spec], axis=2)
				
				new_spec = new_spec.reshape(len(new_spec), np.prod(new_spec.shape[1:]))
				decoded_spectrogram = self.network.model.predict(new_spec)

				keyboard_out, mallet_out = np.split(decoded_spectrogram[0], 2)

				all_mallet.append(mallet_out)

		# pdb.set_trace()

		mallet_so_far = []
		for mallet in all_mallet:
			mallet = mallet.reshape(1,257,251,2)
			mallet_so_far.append(mallet[:,:,:x,:])

		output = np.concatenate(mallet_so_far, axis=2)
		if 251 > output.shape[2]:
			zero_spec = np.zeros((1,257,251 - output.shape[2],2))
			output = np.concatenate([output, zero_spec], axis=2)
			output = output.reshape(len(output), np.prod(output.shape[1:]))

		# pdb.set_trace()

		test_data.spectrogram_to_wav(
				filename=filename.replace('keyboard','mallet', 2),
				spectrogram=copy.deepcopy(mallet_out),
				)
		
		test_data.visualize(
				filename=filename,
				spectrogram = new_spec)

		test_data.visualize(
				filename=filename.replace('keyboard','mallet', 2),
				spectrogram = mallet_out)


	def test_orig(self):

		for filename in self.params['test_data_path']:

			print(filename)
			test_data = Spectrogram(filenames=[filename])

			decoded_spectrogram = self.network.model.predict(test_data.spectrogram)

			keyboard_out, mallet_out = np.split(decoded_spectrogram[0], 2)

		test_data.spectrogram_to_wav(
				filename=filename.replace('keyboard','mallet', 2),
				spectrogram=copy.deepcopy(mallet_out),
				)
		test_data.visualize(
				filename=filename.replace('keyboard','mallet', 2),
				spectrogram = mallet_out)

	def test_long(self):

		for filename in self.params['test_data_path']:

			stride = 35
			n_freq = 257
			n_channel = 2
			spec = Spectrogram(filenames=[filename])
			full_spec = spec.spectrogram.reshape(1, n_freq, -1, n_channel)

			n_time = full_spec.shape[2]

			recon_spec = np.zeros(full_spec.shape)
			output_spec = np.zeros(full_spec.shape)
			count = np.zeros(full_spec.shape)

			count_update_window = np.ones((1, n_freq, 251, n_channel))

			idx = 0
			while True:
				start = stride * idx
				end = start + 251

				idx += 1

				print start

				if end > n_time:
					break

				cur_window = full_spec[:, :, start: end, :]
				cur_window1 = cur_window.reshape(1, -1)

				decoded_spectrogram = self.network.model.predict(cur_window1)

				recon_window, out_window = np.split(decoded_spectrogram[0], 2)



				out_window = out_window.reshape(1, n_freq, 251, n_channel)
				recon_window = recon_window.reshape(1, n_freq, 251, n_channel)

				output_spec[:, :, start: end, :] = np.maximum(out_window, output_spec[:, :, start: end, :])
				# output_spec[:, :, start: end, :] += out_window
				recon_spec[:, :, start: end, :] = np.maximum(recon_window, recon_spec[:, :, start: end, :])
				# recon_spec[:, :, start: end, :] += recon_window
				count[:, :, start: end, :] += count_update_window

				spec.visualize(
					filename=filename,
					spectrogram = copy.deepcopy(output_spec))


				assert count[:, :, start: end, :].shape == count_update_window.shape

			# print count
			# pdb.set_trace()

			count[count == 0] = 1

			# output_spec /= count
			# recon_spec /= count

			spec.spectrogram_to_wav(
					filename=filename,
					spectrogram=copy.deepcopy(recon_spec),
					outfile='reconstruction.wav'
					)

			spec.spectrogram_to_wav(
					filename=filename,
					spectrogram=copy.deepcopy(output_spec),
					outfile='output.wav'
					)
			
			spec.visualize(
					filename=filename,
					spectrogram = recon_spec)

			spec.visualize(
					filename=filename.replace('keyboard','mallet', 2),
					spectrogram = output_spec)


	def load_network(self):
		if eval(self.params['load_weights']):
			model_load_file = self.trained_weights_path + 'model_weights'
			load_ckpt_number = self.params['load_ckpt_number']
			if load_ckpt_number != 0:
				model_load_file += '-0-' + str(load_ckpt_number)
			model_load_file += '.h5'

			self.network.load_model_weights(model_load_file)
