from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sys, os
import numpy as np

import pdb

class Spectrogram(object):

	def __init__(self, filenames = None, spectrogram = None):
		pass
		self.max_const = 1e7
		if filenames == None:
			self.spectrogram_to_wav(spectrogram)
		elif spectrogram == None:
			self.wav_to_spectrogram(filenames)

	def wav_to_spectrogram(self, filenames):

		spectrogram = []
		for fname in filenames:
			sample_rate, samples = wavfile.read(fname)
			frequencies, times, sgram = signal.spectrogram(samples, sample_rate)
			spectrogram.append(sgram)

		spectrogram = np.array(spectrogram) / self.max_const
		self.spectrogram = spectrogram.reshape(spectrogram.shape + (1,))

	def spectrogram_to_wav(self, spectrogram):

		spectrogram *= self.max_const

	def visualize(self, filename, spectrogram=None):
		print filename
		sample_rate, samples = wavfile.read(filename)
		frequencies, times, sgram = signal.spectrogram(samples, sample_rate)

		ax = plt.subplot(121)
		plt.pcolormesh(times, frequencies, sgram)
		plt.imshow(sgram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		ax.set_title("input")

		spectrogram *= self.max_const
		spectrogram = np.reshape(spectrogram, (len(frequencies), len(times)))

		ax = plt.subplot(122)
		plt.pcolormesh(times, frequencies, spectrogram)
		plt.imshow(spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		ax.set_title('output')
		plt.suptitle("filename: {}".format(filename))
		plt.show()
