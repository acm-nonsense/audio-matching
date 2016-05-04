import numpy as np
import matplotlib.pyplot as plt
import sys

figure = plt.figure()

ssm_file = open('ssm.npz', 'r')
ssm = np.load(ssm_file)
n_pts = int(sys.argv[1])


def smooth_array(arr,n_pts):
	smoothed_arr = distances
	for i in range(len(arr)):
		if i >= n_pts/2 and i < len(arr) - n_pts/2:
			total = 0
			for j in range(n_pts):
				total += arr[i+j-n_pts/2]
			smoothed_arr[i] = total/float(n_pts)
		else:
			smoothed_arr[i] = arr[i]
	return smoothed_arr


# select top parts or something


distances = smooth_array(ssm,n_pts)
plt.plot(distances)
figure.savefig('../distances-{}.png'.format(n_pts))
plt.close()