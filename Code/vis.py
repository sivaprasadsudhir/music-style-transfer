from spectrogram import Spectrogram

import os

# dataset_path = '../nsynth-train/guitar/'
# file_list = os.listdir(dataset_path)
# file_list = [os.path.join(dataset_path, i) for i in file_list]

def get_file(instrument, pitch, index = "000", velocity = "050"):
	return "../Data/" + instrument + "/" + instrument + "_" \
			+ index + "-" + pitch + "-" + velocity + ".wav"



guitar = [get_file("guitar_acoustic", i) for i in ["055", "060", "080", "100"]]
keyboard = [get_file("keyboard_acoustic", i) for i in ["055", "060", "080", "100"]]

data = Spectrogram(filenames=keyboard)

# for f in keyboard:
# 	data.visualize(f)
# 	print "Completed:", f

print keyboard[0]
data.spectrogram_to_wav(data.spectrogram[0])
