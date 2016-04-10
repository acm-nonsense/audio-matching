import sys
import numpy as np
from sklearn.decomposition import RandomizedPCA
from time import time
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	print("usage: python cluster_spectrograms.py spectrogram-file-name")
	sys.exit(1)

specs_file = open('{}.specs.npz'.format(sys.argv[1]), 'r')
spectrograms = np.load(specs_file)
bins_file = open('{}.bins.npz'.format(sys.argv[1]), 'r')
bins = np.load(bins_file)
freqs_file = open('{}.freqs.npz'.format(sys.argv[1]), 'r')
freqs = np.load(freqs_file)

plt.pcolormesh(bins, freqs, 10 * np.log10(spectrograms[0]))

n_samples = spectrograms.shape[0]
h = spectrograms.shape[1]
w = spectrograms.shape[2]

data = spectrograms.reshape((n_samples,h*w))
n_components = 15

print("Extracting the top %d eigenfaces from %d faces..." % (n_components, n_samples))
t0 = time()
pca = RandomizedPCA(n_components=n_components, whiten=True).fit(data)
print("Done in %0.3fs." % (time() - t0))

eigensounds = pca.components_.reshape((n_components, h, w))

# figure = plt.figure()
# plt.imshow(eigensounds[0])
# figure.savefig('./eigen.png')

for i in range(n_components):
	figure = plt.figure()
	# plt.imshow(spectrograms[0])
	# figure.savefig('./spec.png')
	plt.pcolormesh(bins, freqs, 10 * np.log10(abs(eigensounds[i])))
	plt.axis('tight')
	figure.savefig('./eigen-{}.png'.format(i))
	plt.close()