import sys
import numpy as np
from numpy.linalg import norm
from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import NearestNeighbors as NN
from sklearn.cluster import MiniBatchKMeans
from time import time
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

if len(sys.argv) < 2:
	print("usage: python cluster_spectrograms.py spectrogram-file-name")
	sys.exit(1)

specs_file = open('{}.specs.npz'.format(sys.argv[1]), 'r')
spectrograms = np.load(specs_file)
bins_file = open('{}.bins.npz'.format(sys.argv[1]), 'r')
bins = np.load(bins_file)
freqs_file = open('{}.freqs.npz'.format(sys.argv[1]), 'r')
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
figure.savefig('../similarity_matrix.png')
plt.close()