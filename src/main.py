from spectrogram import Spectrogram

def get_file_name(instrument, pitch, index = "000", velocity = "050"):
	return "../nsynth-train/" + instrument + "/" + instrument + "_" \
			+ index + "-" + pitch + "-" + velocity + ".wav"

guitar = [get_file_name("guitar_acoustic", i) for i in \
				["055", "060", "080", "100"]]
keyboard = [get_file_name("keyboard_acoustic", i) for i in \
				["055", "060", "080", "100"]]

for f in guitar + keyboard:
	d = Spectrogram(f)
	d.Visualize("random")