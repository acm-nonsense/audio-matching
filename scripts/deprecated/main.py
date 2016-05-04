import json
from scipy.io.wavfile import read
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
import os


from numpy.linalg import norm
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans
from time import time
import matplotlib.pyplot as plt


WINDOW_LENGTH_SECONDS = 5
WINDOW_INTERVAL = 1
SAMPLE_RATE = 44100
WINDOW_LENGTH_SAMPLES = WINDOW_LENGTH_SECONDS * SAMPLE_RATE


def generate_spectrograms(source_file):
	(fs, x) = read("../samples/{}".format(source_file))


	channels = [
		np.array(x[:, 0]),
		np.array(x[:, 1])
	]

	# Combine channels to make a mono signal out of stereo
	channel =  np.mean(channels, axis=0)


	N_SAMPLES = np.shape(channel)[0]
	LENGTH_SECONDS = N_SAMPLES/SAMPLE_RATE



	channel_windows = []
	for i in range(LENGTH_SECONDS/WINDOW_INTERVAL-WINDOW_LENGTH_SECONDS):
		channel_windows.append(channel[SAMPLE_RATE*i:SAMPLE_RATE*(i+WINDOW_LENGTH_SECONDS)])

	print("Created channel windows.")

	print 'Generating spectrograms: ',


	SPECTRUM_HEIGHT = 129
	spectrograms = None
	for i,window in enumerate(channel_windows):
		# figure = plt.figure()
		spectrum, freqs, bins, plot = pylab.specgram(
			window,
			NFFT=256,
			Fs=44100,
			detrend=pylab.detrend_none,
			window=pylab.window_hanning,
			noverlap=int(256 * 0.5))
		if spectrograms == None:
			spectrograms = np.ndarray((0,SPECTRUM_HEIGHT,spectrum.shape[1]))
		if i == 1:
			bins_static = bins
			freqs_static = freqs
		# plt.pcolormesh(bins, freqs, 10*np.log10(spectrum))
		# figure.savefig('../spectrograms/anb-{}.png'.format(i))
		full_dim_spectrum = np.ndarray((1,SPECTRUM_HEIGHT,spectrum.shape[1]))
		# if np.shape(spectrograms)[1:3] == np.shape(spectrum)[0:2]:
		full_dim_spectrum[0] = spectrum[:129,:1721]
		spectrograms = np.concatenate((spectrograms,full_dim_spectrum))
		# plt.close()
		print "{0:2.0f}%\b\b\b\b".format(100*float(i)/len(channel_windows)),
		sys.stdout.flush()
	specs = open('../temp/{}.specs.npz'.format(source_file), 'w')
	bins = open('../temp/{}.bins.npz'.format(source_file), 'w')
	freqs = open('../temp/{}.freqs.npz'.format(source_file), 'w')
	np.save(specs,spectrograms)
	np.save(bins,bins_static)
	np.save(freqs,freqs_static)
	print 'Done.'



def compute_distance(source_file,target_file,sampling_window_index):
	specs_file = open('../temp/{}.specs.npz'.format(source_file), 'r')
	spectrograms = np.load(specs_file)

	target_specs_file = open('../temp/{}.specs.npz'.format(target_file), 'r')
	target_spectrograms = np.load(target_specs_file)

	# pca shit
	n_samples = target_spectrograms.shape[0]
	h = target_spectrograms.shape[1]
	w = target_spectrograms.shape[2]

	data = target_spectrograms.reshape((n_samples,h*w))
	n_components = 100

	# print("Extracting the top %d eigensounds from %d windows..." % (n_components, n_samples))
	# t0 = time()
	# pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data)
	# print("Done in %0.3fs." % (time() - t0))

	# figure = plt.figure()
	# plt.plot(map(lambda v:  norm(v), pca.components_))
	# figure.savefig('../pca.png')
	# plt.close()

	# eigensounds = pca.components_.reshape((n_components, h, w))

	# figure = plt.figure()
	# plt.imshow(eigensounds[0])
	# figure.savefig('./eigen.png')

	# print("Drawing eigensound spectrograms...")
	# for i in range(n_components):
	# 	figure = plt.figure()
	# 	# plt.imshow(spectrograms[0])
	# 	# figure.savefig('./spec.png')
	# 	plt.pcolormesh(bins, freqs, 10 * np.log10(abs(eigensounds[i])))
	# 	plt.axis('tight')
	# 	figure.savefig('./eigen-{}.png'.format(i))
	# 	plt.close()

	# print("Projecting the input data on the eigensounds...")
	# t0 = time()
	# pca_projected_data = pca.transform(data)
	# print("Done in %0.3fs." % (time() - t0))

	n_clusters = 60

	# cluster the sources file windows
	print("Computing clustering for each PCA projected window...")
	print(spectrograms.shape)
	# sys.exit()
	t0 = time()
	source_clusterings = map(lambda sample: MiniBatchKMeans(n_clusters=n_clusters).fit(sample).cluster_centers_, spectrograms)
	print("Done in %0.3fs." % (time() - t0))
	sample_window_clustering = source_clusterings[sampling_window_index]

	# cluster the target file windows
	print("Computing clustering for each PCA projected window...")
	print(target_spectrograms.shape)
	# sys.exit()
	t0 = time()
	target_clusterings = map(lambda sample: MiniBatchKMeans(n_clusters=n_clusters).fit(sample).cluster_centers_, target_spectrograms)
	print("Done in %0.3fs." % (time() - t0))


	print("Computing closest window to specified target window...")
	t0 = time()
	distances_to_target_clusterings = map(lambda current_target_window_clustering: norm(current_target_window_clustering - sample_window_clustering), target_clusterings)
	print("Done in %0.3fs." % (time() - t0))

	# indexed_distances = np.stack((distances_to_target_clustering,np.arange(len(distances_to_target_clustering))),1)
	figure = plt.figure()
	plt.plot(distances_to_target_clusterings)


	# get rid of slashes and shit
	target_file = target_file.split("/")[-1]

	figure.savefig('../results/distances_from_{}_at_{}_to_{}.png'.format(source_file,sampling_window_index,target_file))
	# plt.show()
	plt.close()





def main():
	global WINDOW_LENGTH_SECONDS
	global WINDOW_INTERVAL

	if len(sys.argv) not in (5,7):
		print("usage: python {} source_file.wav sampling-window-index -f/-d target_file.wav/target_directory [window-length window-interval]".format(os.path.basename(__file__)))
		sys.exit(1)

	if len(sys.argv) == 6:
		WINDOW_LENGTH_SECONDS = sys.argv[5]
		WINDOW_INTERVAL = sys.argv[6]
		pass

	source_file = sys.argv[1]
	sampling_window_index = int(sys.argv[2])
	target_type = sys.argv[3]
	target_name = sys.argv[4]

	# generate spectrogram of source file
	generate_spectrograms(source_file)

	if target_type == '-f':
		generate_spectrograms(target_name)
		# print('compute_distance({},{},{})'.format(source_file,target_name,sampling_window_index))
		compute_distance(source_file,target_name,sampling_window_index)

	elif target_type == '-d':
		for target_file in os.listdir('../samples/{}'.format(target_name)):
			if not os.path.exists("../temp/{}/".format(target_name)):
			    os.makedirs("../temp/{}/".format(target_name))

			# print "{}/{}".format(target_name,target_file)
			generate_spectrograms("{}/{}".format(target_name,target_file))
			# print('compute_distance({},{},{})'.format(source_file,"{}/{}".format(target_name,target_file),sampling_window_index))
			compute_distance(source_file,"{}/{}".format(target_name,target_file),sampling_window_index)



if __name__ == "__main__":
	main()