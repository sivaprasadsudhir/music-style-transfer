from autoencoder import Autoencoder
from spectrogram import Spectrogram
from agent import Agent
from utils import *

import json
import pdb

def load_network(params, network, trained_weights_path):
	model_load_file = trained_weights_path + 'model_weights'
	load_ckpt_number = params['load_ckpt_number']
	if load_ckpt_number != 0:
		model_load_file += '-' + str(load_ckpt_number)
	model_load_file += '.h5'

	network.load_model_weights(model_load_file)

def main(args):
	# Parse command-line arguments.
	args = parse_arguments()

	with open(args.config) as f:
		params = json.load(f)

	trans_from = Autoencoder(load_model = True, load_path = params['translate_from'])
	trans_to = Autoencoder(load_model = True, load_path = params['translate_to'])

	load_network(params, trans_from, params['translate_from'])
	load_network(params, trans_to, params['translate_to'])

	test_data_path = params['test_data_path']
	test_data = Spectrogram(filenames=test_data_path)

	encoded_input_from = trans_from.encoder.predict(test_data.spectrogram)
	
	print (encoded_input_from)
	
	decoded_spectrogram = trans_from.decoder.predict(encoded_input_from)
	test_data.visualize(filename=test_data_path[0],
			                    spectrogram = decoded_spectrogram)

	decoded_full_spectrogram = trans_from.model.predict(test_data.spectrogram)

	test_data.visualize(filename=test_data_path[0],
			                    spectrogram = decoded_full_spectrogram)
	
	print (decoded_spectrogram)
	print (decoded_full_spectrogram)

	# test_data_path[0] = test_data_path[0].replace('mallet', 'keyboard', 2)
	# test_data = Spectrogram(filenames=test_data_path)

	# encoded_input_to = trans_to.encoder.predict(test_data.spectrogram)
	# print (encoded_input_to)

	# decoded_spectrogram = trans_from.decoder.predict(encoded_input_to)
	# test_data.visualize(filename=test_data_path[0],
	# 		                    spectrogram = decoded_spectrogram)
	
if __name__ == '__main__':
	main(sys.argv)
