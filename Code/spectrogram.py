from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sys, os
import numpy as np

import pdb

class Spectrogram(object):

	def __init__(self, filenames = None, spectrogram = None):

		self.max_const = 10e7
		if filenames == None:
			self.spectrogram_to_wav(spectrogram)
		elif spectrogram == None:
			self.wav_to_spectrogram(filenames)

	def wav_to_spectrogram(self, filenames):
		
		spectrogram = []
		for fname in filenames:
			sample_rate, samples = wavfile.read(fname)
			frequencies, times, sgram = signal.spectrogram(samples, sample_rate)

			# pdb.set_trace()
			spectrogram.append(sgram)

		spectrogram = np.array(spectrogram) / self.max_const
		self.spectrogram = spectrogram.reshape((len(spectrogram), np.prod(spectrogram.shape[1:])))

	def spectrogram_to_wav(self, spectrogram):

		spectrogram *= self.max_const

	def visualize(self, filename, spectrogram):
		
		sample_rate, samples = wavfile.read(filename)
		frequencies, times, sgram = signal.spectrogram(samples, sample_rate)

		spectrogram = np.reshape(spectrogram, (len(frequencies), len(times)))

		plt.pcolormesh(times, frequencies, sgram)
		plt.imshow(sgram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.savefig('original.png')

		plt.pcolormesh(times, frequencies, spectrogram)
		plt.imshow(spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		plt.savefig('transformed.png')


