from features import mfcc
from features import logfbank

import scipy.io.wavfile as wav
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import sys
from time import time
import warnings

# actually, we're not computing the cross-dot=products, just the diagonals (sampling_window_length sums)... so maybe we can try doing that 
# Should take an audio file, compute the mfcc (which is basically a feature vector), then computes the similarity matrix with square dimensions (dim(mfcc)/sampling_window_length)-squared by averaging over every cross=dot=product for a window of size sampling_window_length
# thus, there are sampling_window_length*sampling_window_length sums

#w = 10
sampling_window_length = 10
audio_file = sys.argv[1]
file_head_length = 0 # in seconds

warnings.filterwarnings("ignore", category=wav.WavFileWarning)

def compute_mfcc():
    print("\tComputing MFCCs...")
    t0 = time()
    (rate,sig) = wav.read(audio_file)
    if file_head_length != 0: # just take the first file_head_length seconds of the audio file
        sig = sig[0:rate*file_head_length,:] # delete this after testing
    mfcc_feat = mfcc(sig,rate)
    #fbank_feat = logfbank(sig,rate) # potentially look at this later?
    print("\tDone in %0.3fs." % (time() - t0))
    return mfcc_feat

def compute_similarity_matrix(mfcc_feat):
    print("\tComputing similarity matrix...")
    t0 = time()
    sig_length = len(mfcc_feat)
    similarity_matrix = np.ndarray((sig_length/sampling_window_length,sig_length/sampling_window_length))
    for i in range(sig_length/sampling_window_length):
            for j in range(sig_length/sampling_window_length):
                    entry = 0
                    for k in range(sampling_window_length):
                            entry += np.dot(mfcc_feat[i*sampling_window_length+k,:],mfcc_feat[j*sampling_window_length+k,:])/(la.norm(mfcc_feat[i*sampling_window_length+k,:])*la.norm(mfcc_feat[j*sampling_window_length+k,:]))
                            #entry += np.dot(mfcc_feat[i*sampling_window_length+k,:],mfcc_feat[j*sampling_window_length+k,:])
                    similarity_matrix[i:i+sampling_window_length,j:j+sampling_window_length] = entry/float(sampling_window_length) 
                    print "\t{0:2.0f}%\b\b\b\b\b".format(100*float(i*sampling_window_length*sig_length+j*sampling_window_length)/float(sig_length*sig_length)),
                    # print(i*wlen(mfcc_feat)+j*sampling_window_length)
                    # print(float(len(mfcc_feat)*len(mfcc_feat)))
                    sys.stdout.flush()
    print("\tDone in %0.3fs." % (time() - t0))
    return similarity_matrix
                    
def standardize_matrix(input_matrix):
    print("\tStandardizing similarity matrix...")
    t0 = time()
    matrix_mean = mean(input_matrix)
    matrix_std = std(input_matrix)
    input_matrix = (input_matrix-matrix_mean)/matrix_std
    print("\tDone in %0.3fs." % (jime() - t0))
    return input_matrix

def normalize_matrix(input_matrix):
    print("\tNormalizing similarity matrix...")
    t0 = time()
    max = input_matrix.max()
    min = input_matrix.min()
    input_matrix -= min
    input_matrix /= (max-min)
    print("\tDone in %0.3fs." % (time() - t0))
    return input_matrix

def save_matrix(input_matrix):
    print("\tVisualizing similarity matrix...")
    t0 = time()
    # are we losing information if we try to cluster directly on the images? http://stackoverflow.com/questions/24185083/change-resolution-of-imshow-in-ipython
    #figure = plt.figure(figsize = (44100*sampling_window_length/1000,44100*sampling_window_length/1000))
    figure = plt.figure()
    plt.title(audio_file)
    plt.imshow(input_matrix, cmap="gray",interpolation='none',origin='lower')
    #plt.show()
    #figure.savefig(audio_file[:-4]+"_sampling_window_length_"+str(sampling_window_length)+".png")
    figure.savefig(audio_file[:-4]+"_sampling_window_length_"+str(sampling_window_length)+"_head_length_"+str(file_head_length)+".png")
    print("\tDone in %0.3fs." % (time() - t0))

def main():
    mfcc_feat = compute_mfcc()
    similarity_matrix = compute_similarity_matrix(mfcc_feat)
    normalized_matrix = normalize_matrix(similarity_matrix)
    save_matrix(normalized_matrix)
    
if __name__ == "__main__":
    main()
