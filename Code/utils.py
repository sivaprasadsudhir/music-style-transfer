import argparse
import sys, os, errno

def create_folder(directory):
	try:
		os.makedirs(directory)
	except OSError as e:
		if e.errno != errno.EEXIST:
			raise

def parse_arguments():
	# Command-line flags are defined here.
	parser = argparse.ArgumentParser()
	parser.add_argument('config', help="location of the config file")
	return parser.parse_args()


# if we need to find matching samples between two instruments again,
# the following code can be tweaked to do so

# def instrument_shared_samples():
# 	file_list_original = os.listdir(self.params['data_path'])
# 	file_list_original = [os.path.join(self.params['data_path'], i)
# 			for i in file_list_original[:]]
#
# 	mallet_file_list = [filename.replace("keyboard_", "mallet_", 2)
# 						for filename in file_list_original]
#
# 	mask = [i for i, filename in enumerate(mallet_file_list)
# 			if os.path.isfile(filename)]
#
# 	file_list = [file_list_original[i] for i in mask]
