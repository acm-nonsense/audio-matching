import sys
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import RandomizedPCA
from sklearn.cluster import MiniBatchKMeans
from time import time
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	print("usage: python cluster_spectrograms.py spectrogram-file-name target-offset")
	sys.exit(1)

specs_file = open('../temp/{}.specs.npz'.format(sys.argv[1]), 'r')
spectrograms = np.load(specs_file)
bins_file = open('../temp/{}.bins.npz'.format(sys.argv[1]), 'r')
bins = np.load(bins_file)
freqs_file = open('../temp/{}.freqs.npz'.format(sys.argv[1]), 'r')
freqs = np.load(freqs_file)

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
print(spectrograms.shape)
# sys.exit()
t0 = time()
clusterings = map(lambda sample: MiniBatchKMeans(n_clusters=n_clusters).fit(sample).cluster_centers_, spectrograms)
print("Done in %0.3fs." % (time() - t0))


target_window_index = int(sys.argv[2])

print("Computing closest window to specified target window...")
t0 = time()
target_clustering = clusterings[target_window_index]
distances_to_target_clustering = map(lambda sample_clustering: norm(sample_clustering - target_clustering), clusterings)
print("Done in %0.3fs." % (time() - t0))

# indexed_distances = np.stack((distances_to_target_clustering,np.arange(len(distances_to_target_clustering))),1)
figure = plt.figure()
plt.plot(distances_to_target_clustering)
figure.savefig('../distances.png')
plt.close()