from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import sys, os
import numpy as np

class Spectrogram(object):

	'''
	norm_factor
	filename
	sample_rate
	samples
	frequencies
	times
	spectrogram
	'''

	# TODO[Siva]: Do we need to store the times and frequencies?

	def __init__(self, filename=None):
		self.norm_factor = 10e7
		if filename != None:
			self.filename = filename
			self.sample_rate, self.samples = wavfile.read(filename)
			# TODO[Siva]: Change this to a better spectrogram generator
			self.frequencies, self.times, self.spectrogram = \
					signal.spectrogram(self.samples, self.sample_rate)
			self.spectrogram = self.spectrogram * 1.0 / self.norm_factor

	def GenSpectrogram(self, filename):
		self.filename = filename
		self.sample_rate, self.samples = wavfile.read(filename)
		# TODO[Siva]: Change this to a better spectrogram generator
		self.frequencies, self.times, self.spectrogram = \
				signal.spectrogram(self.samples, self.sample_rate)
		self.spectrogram = self.spectrogram * 1.0 / self.norm_factor

	def GenWav(self):
		pass

	def Visualize(self, directory=None):
		plt.pcolormesh(self.times, self.frequencies, self.spectrogram)
		plt.imshow(self.spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')

		if directory == None:
			plt.show()
		else:
			f = self.filename.split('/')[-1]
			os.system("mkdir -p ../" + directory + "/" + f[:-16] + "/")
			plt.savefig('../' + directory + "/" + f[:-16] + "/" + f[:-4] + \
						'.png')


