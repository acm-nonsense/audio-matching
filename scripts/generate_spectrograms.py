import json
from scipy.io.wavfile import read
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys
from numpy.linalg import norm
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.cluster import MiniBatchKMeans
from time import time
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


if len(sys.argv) not in (2,4):
	print("usage: python generate_spectrograms.py audio.wav window-length window-interval\n\
		or default to 5;1: python generate_spectrograms.py audio.wav")
	sys.exit(1)
# elif len(sys.argv) < 4:
# 	print("usage: python generate_spectrograms.py audio.wav window-length window-interval")
# 	sys.exit(1)

# Read file to get samplerate and numpy array containing the signal 
(fs, x) = read(sys.argv[1])


channels = [
	np.array(x[:, 0]),
	np.array(x[:, 1])
]

# Combine channels to make a mono signal out of stereo
channel =  np.mean(channels, axis=0)


N_SAMPLES = np.shape(channel)[0]
SAMPLE_RATE = 44100
LENGTH_SECONDS = N_SAMPLES/SAMPLE_RATE

WINDOW_LENGTH_SECONDS = 5
WINDOW_INTERVAL = 1

if len(sys.argv) == 4:
	WINDOW_LENGTH_SECONDS = int(sys.argv[2])
	WINDOW_INTERVAL = int(sys.argv[3]) # In seconds

WINDOW_LENGTH_SAMPLES = WINDOW_LENGTH_SECONDS * SAMPLE_RATE

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
# specs = open('{}.specs.npz'.format(sys.argv[1]), 'w')
# bins = open('{}.bins.npz'.format(sys.argv[1]), 'w')
# freqs = open('{}.freqs.npz'.format(sys.argv[1]), 'w')
# np.save(specs,spectrograms)
# np.save(bins,bins_static)
# np.save(freqs,freqs_static)
print 'Done.'

# specs_file = open('{}.specs.npz'.format(sys.argv[1]), 'r')
# spectrograms = specs
# bins_file = open('{}.bins.npz'.format(sys.argv[1]), 'r')
bins = bins_static
# freqs_file = open('{}.freqs.npz'.format(sys.argv[1]), 'r')
freqs = freqs_static

n_samples = spectrograms.shape[0]
h = spectrograms.shape[1]
w = spectrograms.shape[2]

data = spectrograms.reshape((n_samples,h*w))
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

print("Computing clustering for each PCA projected window...")
# sys.exit()
t0 = time()
clusterings = map(lambda sample: MiniBatchKMeans(n_clusters=n_clusters).fit(sample).cluster_centers_, spectrograms)
print("Done in %0.3fs." % (time() - t0))

print("Computing audio self similarity matrix...")
t0 = time()
similarity_matrix = np.ndarray((len(clusterings),len(clusterings)))
for target_index in range(len(clusterings)):
	target_clustering = clusterings[target_index]
	neighbors = NN(n_neighbors=1).fit(target_clustering)

	for i in range(len(clusterings)):
		# dists, indxs = neighbors.kneighbors(clusterings[i])
		# similarity_matrix[target_index,i] = norm(dists)
		similarity_matrix[target_index,i] = norm(clusterings[i] - target_clustering)

target_clustering = clusterings[30]
distances_to_target_clustering = map(lambda sample_clustering: norm(sample_clustering - target_clustering), clusterings)
print("Done in %0.3fs." % (time() - t0))


print("Patching and normalizing self similarity matrix...")
t0 = time()
max = similarity_matrix.max()
min = similarity_matrix.min()
similarity_matrix -= min
similarity_matrix /= (max-min)
print("Done in %0.3fs." % (time() - t0))


# indexed_distances = np.stack((distances_to_target_clustering,np.arange(len(distances_to_target_clustering))),1)
figure = plt.figure()
dists_file = open('distances.npz', 'w')
np.save(dists_file,distances_to_target_clustering)
# plt.plot(distances_to_target_clustering)
# plt.show()
plt.imshow(similarity_matrix, cmap="inferno",interpolation='none')
figure.savefig(sys.argv[1][:-4]+".png")
plt.close()
