from matplotlib import pyplot as plt
from scipy import signal
from scipy.io import wavfile
import librosa
import numpy as np
import os, sys, pdb
import tensorflow as tf

import pdb

def trim_for_encoding(wav_data, sample_length, hop_length=512):
	"""Make sure audio is a even multiple of hop_size.

	Args:
	wav_data: 1-D or 2-D array of floats.
	sample_length: Max length of audio data.
	hop_length: Pooling size of WaveNet autoencoder.

	Returns:
	wav_data: Trimmed array.
	sample_length: Length of trimmed array.
	"""
	if wav_data.ndim == 1:
		# Max sample length is the data length
		if sample_length > wav_data.size:
			sample_length = wav_data.size
			# Multiple of hop_length
			sample_length = (sample_length // hop_length) * hop_length
			# Trim
			wav_data = wav_data[:sample_length]
			# Assume all examples are the same length
	elif wav_data.ndim == 2:
		# Max sample length is the data length
		if sample_length > wav_data[0].size:
			sample_length = wav_data[0].size
			# Multiple of hop_length
			sample_length = (sample_length // hop_length) * hop_length
			# Trim
			wav_data = wav_data[:, :sample_length]

	return wav_data, sample_length

def specgram(audio,
			 n_fft=512,
			 hop_length=None,
			 mask=True,
			 log_mag=True,
			 re_im=False,
			 dphase=True,
			 mag_only=False):
	if not hop_length:
		hop_length = int(n_fft / 2.)

	fft_config = dict(
		n_fft=n_fft, win_length=n_fft, hop_length=hop_length, center=True)

	spec = librosa.stft(audio, **fft_config)

	if re_im:
		re = spec.real[:, :, np.newaxis]
		im = spec.imag[:, :, np.newaxis]
		spec_real = np.concatenate((re, im), axis=2)

	else:
		mag, phase = librosa.core.magphase(spec)
		phase_angle = np.angle(phase)

		# Magnitudes, scaled 0-1
		if log_mag:
			mag = (librosa.power_to_db(
				mag**2, amin=1e-13, top_db=120., ref=np.max) / 120.) + 1
		else:
			mag /= mag.max()

		if dphase:
			#  Derivative of phase
			phase_unwrapped = np.unwrap(phase_angle)
			p = phase_unwrapped[:, 1:] - phase_unwrapped[:, :-1]
			p = np.concatenate([phase_unwrapped[:, 0:1], p], axis=1) / np.pi
		else:
			# Normal phase
			p = phase_angle / np.pi
		# Mask the phase
		if log_mag and mask:
			p = mag * p
		# Return Mag and Phase
		p = p.astype(np.float32)[:, :, np.newaxis]
		mag = mag.astype(np.float32)[:, :, np.newaxis]
		if mag_only:
			spec_real = mag[:, :, np.newaxis]
		else:
			spec_real = np.concatenate((mag, p), axis=2)
	return spec_real

def inv_magphase(mag, phase_angle):
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	return mag * phase

def griffin_lim(mag, phase_angle, n_fft, hop, num_iters):
	"""Iterative algorithm for phase retrival from a magnitude spectrogram.

	Args:
	mag: Magnitude spectrogram.
	phase_angle: Initial condition for phase.
	n_fft: Size of the FFT.
	hop: Stride of FFT. Defaults to n_fft/2.
	num_iters: Griffin-Lim iterations to perform.

	Returns:
	audio: 1-D array of float32 sound samples.
	"""
	fft_config = dict(n_fft=n_fft, win_length=n_fft, hop_length=hop, center=True)
	ifft_config = dict(win_length=n_fft, hop_length=hop, center=True)
	complex_specgram = inv_magphase(mag, phase_angle)
	for i in range(num_iters):
		audio = librosa.istft(complex_specgram, **ifft_config)
		if i != num_iters - 1:
			complex_specgram = librosa.stft(audio, **fft_config)
			_, phase = librosa.magphase(complex_specgram)
			phase_angle = np.angle(phase)
			complex_specgram = inv_magphase(mag, phase_angle)
	return audio


def ispecgram(spec,
              n_fft=512,
              hop_length=None,
              mask=True,
              log_mag=True,
              re_im=False,
              dphase=True,
              mag_only=True,
              num_iters=1000):
	"""Inverse Spectrogram using librosa.

	Args:
	spec: 3-D specgram array [freqs, time, (mag_db, dphase)].
	n_fft: Size of the FFT.
	hop_length: Stride of FFT. Defaults to n_fft/2.
	mask: Reverse the mask of the phase derivative by the magnitude.
	log_mag: Use the logamplitude.
	re_im: Output Real and Imag. instead of logMag and dPhase.
	dphase: Use derivative of phase instead of phase.
	mag_only: Specgram contains no phase.
	num_iters: Number of griffin-lim iterations for mag_only.

	Returns:
	audio: 1-D array of sound samples. Peak normalized to 1.
	"""
	if not hop_length:
		hop_length = n_fft // 2

	ifft_config = dict(win_length=n_fft, hop_length=hop_length, center=True)

	if mag_only:
		mag = spec[:, :, 0]
		phase_angle = np.pi * np.random.rand(*mag.shape)
	elif re_im:
		spec_real = spec[:, :, 0] + 1.j * spec[:, :, 1]
	else:
		mag, p = spec[:, :, 0], spec[:, :, 1]
		if mask and log_mag:
			p /= (mag + 1e-13 * np.random.randn(*mag.shape))
		if dphase:
			# Roll up phase
			phase_angle = np.cumsum(p * np.pi, axis=1)
		else:
			phase_angle = p * np.pi

	# Magnitudes
	if log_mag:
		mag = (mag - 1.0) * 120.0
		mag = 10**(mag / 20.0)
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	spec_real = mag * phase

	if mag_only:
		audio = griffin_lim(
	    	mag, phase_angle, n_fft, hop_length, num_iters=num_iters)
	else:
		audio = librosa.core.istft(spec_real, **ifft_config)
	return np.squeeze(audio / audio.max())

def normalize(x, y):
	return x * x + y * y

class Spectrogram(object):

	def __init__(self, filenames = None, spectrogram = None):
		self.max_const = 1e7
		self.n_fft = 512
		self.hop_length = int(self.n_fft/2.)
		self.mask = True
		self.log_mag = True
		self.pad = True
		self.re_im = False
		self.dphase = True
		self.mag_only = False
		self.num_iters = 1000
		self.sample_rate = 16000

		if filenames == None:
			self.spectrogram_to_wav(spectrogram)
		elif spectrogram == None:
			self.wav_to_spectrogram(filenames)

	def wav_to_spectrogram(self, filenames):
		spectrogram = []
		# print (filenames)
		for fname in filenames:
			# print (fname)
			audio, _ = librosa.load(fname, sr=self.sample_rate)
			sgram = specgram(audio)
			# print "Shape of the spectrogram", sgram.shape
			spectrogram.append(sgram)

		# self.spectrogram = np.array(spectrogram) / self.max_const
		self.spectrogram = np.array(spectrogram)


		
		# spectrogram = []
		# for fname in filenames:
		# 	# print(fname)
		# 	sample_rate, samples = wavfile.read(fname)
		# 	frequencies, times, sgram = signal.spectrogram(samples, sample_rate)
		# 	spectrogram.append(sgram)

		# pdb.set_trace()
		# spectrogram = np.array(spectrogram) / self.max_const
		# self.spectrogram = spectrogram.reshape(spectrogram.shape + (1,))
		# self.spectrogram = spectrogram.reshape((len(spectrogram), np.prod(spectrogram.shape[1:])))

	def spectrogram_to_wav(self, filename, spectrogram, outfile="output.wav"):
		dims = spectrogram.shape
		print dims
		print "Inside the function"

		print (filename)
		audio, _ = librosa.load(filename, sr=self.sample_rate)
		sgram = specgram(audio)
		n_freq, n_time, unused_channels = sgram.shape

		print (n_freq, n_time)

		# pdb.set_trace()
		# spec = spectrogram * self.max_const
		# spec = spectrogram.reshape(spectrogram.shape[1:])
		spec = spectrogram.reshape((n_freq, n_time, unused_channels))
		print spec.shape
		print filename
		# if self.pad:
			# spec = tf.concat([spec, tf.zeros([1, dims[1], dims[2]])], 0)
		audio = ispecgram(spec, 
					n_fft=self.n_fft, 
					hop_length=self.hop_length, 
					mask=self.mask, 
					log_mag=self.log_mag, 
					re_im=self.re_im, 
					dphase=self.dphase, 
					mag_only=self.mag_only, 
					num_iters=self.num_iters)

		librosa.output.write_wav(outfile, audio, self.sample_rate,
								 norm=False)

	def visualize(self, filename, spectrogram=None):
		print (filename)
		audio, _ = librosa.load(filename, sr=self.sample_rate)
		sgram = specgram(audio)
		n_freq, n_time, unused_channels = sgram.shape
		frequencies = range(n_freq)
		times = range(n_time)

		sgram = normalize(sgram[:, :, 0], sgram[:, :, 1])
		sgram = np.reshape(sgram, (len(frequencies), len(times)))

		ax = plt.subplot(121)
		plt.pcolormesh(times, frequencies, sgram)
		plt.imshow(sgram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		ax.set_title("input")

		# if spectrogram != None:
		# spectrogram *= self.max_const
		print(spectrogram.shape)
		# pdb.set_trace()
		spectrogram = spectrogram.reshape((1, n_freq, n_time, unused_channels))
		spectrogram = normalize(spectrogram[0, :, :, 0], spectrogram[0, :, :, 1])
		print(spectrogram.shape)
		spectrogram = np.reshape(spectrogram, (len(frequencies),
									len(times)))
		print(spectrogram.shape)
		ax = plt.subplot(122)
		plt.pcolormesh(times, frequencies, spectrogram)
		plt.imshow(spectrogram)
		plt.ylabel('Frequency [Hz]')
		plt.xlabel('Time [sec]')
		ax.set_title('output')

		plt.suptitle("filename: {}".format(filename))
		plt.show()
