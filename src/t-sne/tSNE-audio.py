import argparse
import sys
import fnmatch
import os
import numpy as np
import librosa
import essentia.standard as ess
import tqdm
from sklearn.manifold import TSNE
import json


def process_arguments(args):
	parser = argparse.ArgumentParser(description='tSNE on audio')
	parser.add_argument('--input_dir', action='store', help='path to directory of input files')
	parser.add_argument('--output_file', action='store', help='path to where to store t-SNE analysis in json')
	parser.add_argument('--num_dimensions', action='store', default=2, help='dimensionality of t-SNE points (default 2)')
	parser.add_argument('--perplexity', action='store', default=150, help='perplexity of t-SNE (default 30)')
	params = vars(parser.parse_args(args))
	return params


def get_audio_files(path):
	files = []
	for root, dirnames, filenames in os.walk(path):
		for filename in fnmatch.filter(filenames, '*.wav'):
			files.append(os.path.join(root, filename))
	return files


def is_silent(audio):
	""" returns True if audio is silent (below threshold) """
	rms = np.sqrt(np.mean(np.square(audio)))
	rms_threshold = 10 ** (-60 / 20)
	if rms > rms_threshold:
		return False
	else:
		return True


def iqr(x, axis):
	#NOTE: x can be 2-dimensional (e.g. mfccs)
	q75, q25 = np.percentile(x, [75 ,25], axis=axis)
	iqr = q75 -q25
	return iqr


def get_features(audio, sr):
	""" returns vector with multiple concatenated timbre descriptors """
	#TODO: discard 0th mfcc coefficient (total energy)?
	n_mfccs = 12
	feature_vector = []

	# mfccs
	mfccs = librosa.feature.mfcc(audio, sr=sr, n_mfcc=n_mfccs)
	feature_vector = np.append(feature_vector, np.median(mfccs, axis=1))
	feature_vector = np.append(feature_vector, iqr(mfccs, axis=1))
	
	# other features
	spec_cent = librosa.feature.spectral_centroid(audio, sr=sr)
	zero_cross_rate = librosa.feature.zero_crossing_rate(audio + 0.0001)
	spec_flatness = librosa.feature.spectral_flatness(audio)
	feature_vector = np.append(feature_vector, np.median(spec_cent, axis=1))
	feature_vector = np.append(feature_vector, np.median(zero_cross_rate, axis=1))
	feature_vector = np.append(feature_vector, np.median(spec_flatness, axis=1))
	feature_vector = np.append(feature_vector, iqr(spec_cent, axis=1))
	feature_vector = np.append(feature_vector, iqr(zero_cross_rate, axis=1))
	feature_vector = np.append(feature_vector, iqr(spec_flatness, axis=1))
	
	return feature_vector


# def get_features(audio, sr):
# 	""" returns vector with muliptle concatenated timbre descriptors """
# 	framesize = 2048 #use bigger framesize (better freq resolution) !!!!
# 	hopsize = 512
# 	spec_size = int(framesize / 2 + 1)
# 	n_mfccs = 12

# 	window = ess.Windowing(type='hann', size=framesize) #blackman-harris window for SpectralPeaks()?
# 	spectrum = ess.Spectrum(size=framesize)
# 	spec_peaks = ess.SpectralPeaks(sampleRate=sr, magnitudeThreshold=-60, maxPeaks=100)
# 	yin = ess.PitchYinFFT(frameSize=framesize, sampleRate=sr)
# 	harm_peaks = ess.HarmonicPeaks(maxHarmonics=20, tolerance=0.2)
# 	inharmonicity = ess.Inharmonicity()
# 	odd_to_even = ess.OddToEvenHarmonicEnergyRatio()
# 	tristimulus = ess.Tristimulus()
# 	mfcc = ess.MFCC(inputSize=spec_size, numberCoefficients=n_mfccs, sampleRate=sr)
# 	zero_crossing_rate = ess.ZeroCrossingRate(threshold=0)
# 	spectral_centroid = ess.Centroid(range=sr/2)

# 	#TODO: maybe also compute envelope features like attack-time etc. (essentia --> LogAttackTime)

# 	mfcc_array = []
# 	spec_cent_array = []
# 	zcr_array = []
# 	inharm_array = []
# 	odd_to_even_array = []
# 	tristim_array = []
# 	for frame in ess.FrameGenerator(audio, frameSize=framesize, hopSize=hopsize, startFromZero=True):
# 		if not is_silent(frame):
# 			zcr = zero_crossing_rate(frame)
# 			spec = spectrum(window(frame))
# 			spec_db = 20 * np.log10(spec + 10**-20) #add small number to avoid log(0)
# 			sp_freqs, sp_magn = spec_peaks(spec_db)
# 			f_0, _ = yin(spec)
# 			if len(sp_freqs) > 0:
# 				if sp_freqs[0] == 0:
# 					sp_freqs = sp_freqs[1:]
# 					sp_magn = sp_magn[1:]
# 			harm_freqs, harm_magn = harm_peaks(sp_freqs, sp_magn, f_0) 
# 			#If a particular harmonic was not found among spectral peaks, its ideal frequency value is output together with 0 magnitude
# 			inharm = inharmonicity(harm_freqs, harm_magn)
# 			odd_to_even_harm_energy_ratio = odd_to_even(harm_freqs, harm_magn)
# 			tristim = tristimulus(harm_freqs, harm_magn)
# 			_, mfccs = mfcc(spec)
# 			spec_cent = spectral_centroid(spec)

# 			mfcc_array.append(mfccs)
# 			spec_cent_array.append(spec_cent)
# 			zcr_array.append(zcr)
# 			inharm_array.append(inharm)
# 			odd_to_even_array.append(odd_to_even_harm_energy_ratio)
# 			tristim_array.append(tristim)
# 		else:
# 			print('ssos')

# 	# median instead of mean
# 	mfccs_mean = np.median(mfcc_array, axis=0)
# 	spec_cent_mean = np.median(spec_cent_array, axis=0)
# 	zcr_mean = np.median(zcr_array, axis=0)
# 	inharm_mean = np.median(inharm_array, axis=0)
# 	odd_to_even_mean = np.median(odd_to_even_array, axis=0)
# 	tristim_mean = np.median(tristim_array, axis=0)

# 	# interquartile range
# 	mfccs_iqr = iqr(mfcc_array)
# 	spec_cent_iqr = iqr(spec_cent_array)
# 	zcr_iqr = iqr(zcr_array)
# 	inharm_iqr = iqr(inharm_array)
# 	odd_to_even_iqr = iqr(odd_to_even_array)
# 	tristim_iqr = iqr(tristim_array)


# 	feature_vector = []
# 	#TODO: maybe scale small/big values by a constant (e.g. 1000)
# 	feature_vector = np.append(feature_vector, zcr_mean)
# 	feature_vector = np.append(feature_vector, mfccs_mean)
# 	#feature_vector = np.append(feature_vector, spec_cent_mean)
# 	feature_vector = np.append(feature_vector, inharm_mean)
# 	feature_vector = np.append(feature_vector, odd_to_even_mean)
# 	feature_vector = np.append(feature_vector, tristim_mean)

# 	feature_vector = np.append(feature_vector, zcr_iqr)
# 	feature_vector = np.append(feature_vector, mfccs_iqr)
# 	#feature_vector = np.append(feature_vector, spec_cent_iqr)
# 	feature_vector = np.append(feature_vector, inharm_iqr)
# 	feature_vector = np.append(feature_vector, odd_to_even_iqr)
# 	feature_vector = np.append(feature_vector, tristim_iqr)

# 	#feature_vector = (feature_vector-np.mean(feature_vector)) / np.std(feature_vector)  #NOTE: division by zero if feature_vector contains only zeros

# 	return feature_vector


def analyze_directory(input_dir):	
	files = get_audio_files(input_dir)
	feature_vectors = []
	for file in tqdm.tqdm(files):
		audio, sr = librosa.load(file, sr=None)
		feat = get_features(audio, sr)
		label = file.split('/')[-2]
		feature_vectors.append({"file":file, "features":feat, "label": label})
	return feature_vectors


def run_tSNE(feature_vectors, tsne_path, tsne_dimensions, tsne_perplexity=30):
	X = [f["features"] for f in feature_vectors]
	tsne = TSNE(n_components=tsne_dimensions, learning_rate=200, perplexity=tsne_perplexity, verbose=1, angle=0.1).fit_transform(X)
	data = []
	for i, f in enumerate(feature_vectors):
		point = [ float(tsne[i,k] - np.min(tsne[:,k]))/(np.max(tsne[:,k]) - np.min(tsne[:,k])) for k in range(tsne_dimensions) ]
		data.append({"path":os.path.abspath(f["file"]), "point":point, "label":f["label"]})
	with open(tsne_path, 'w') as outfile:
		json.dump(data, outfile)


if __name__ == '__main__':
	params = process_arguments(sys.argv[1:])
	output_file_path = params['output_file']
	tsne_dimensions = int(params['num_dimensions'])
	tsne_perplexity = int(params['perplexity'])
	if params['input_dir'] is not None:
		input_dir = params['input_dir']
		feature_vectors = analyze_directory(input_dir)
		run_tSNE(feature_vectors, output_file_path, tsne_dimensions, tsne_perplexity)
		print("finished saving %s"%output_file_path)
	else:
		print("Error: no input specified!")
