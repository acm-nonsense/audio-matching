from features import mfcc
from features import logfbank

import scipy.io.wavfile as wav
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import sys
from time import time
import warnings

w = 10
audio_file = sys.argv[1]

warnings.filterwarnings("ignore", category=wav.WavFileWarning)

def compute_mfcc():
    print("Computing MFCCs...")
    t0 = time()
    (rate,sig) = wav.read(audio_file)
    mfcc_feat = mfcc(sig,rate)
    #fbank_feat = logfbank(sig,rate) # potentially look at this later?
    print("Done in %0.3fs." % (time() - t0))
    return mfcc_feat

def compute_similarity_matrix(mfcc_feat):
        print("Computing similarity matrix...")
        t0 = time()
        sig_length = len(mfcc_feat)
        similarity_matrix = np.ndarray((sig_length/w,sig_length/w))
        for i in range(sig_length/w):
                for j in range(sig_length/w):
                        entry = 0
                        for k in range(w):
                                entry += np.dot(mfcc_feat[i*w+k,:],mfcc_feat[j*w+k,:])/(la.norm(mfcc_feat[i*w+k,:])*la.norm(mfcc_feat[j*w+k,:]))
                                #entry += np.dot(mfcc_feat[i*w+k,:],mfcc_feat[j*w+k,:])
                        similarity_matrix[i:i+w,j:j+w] = entry/float(w) 
                        print "{0:2.0f}%\b\b\b\b".format(100*float(i*w*sig_length+j*w)/float(sig_length*sig_length)),
                        # print(i*wlen(mfcc_feat)+j*w)
                        # print(float(len(mfcc_feat)*len(mfcc_feat)))
                        sys.stdout.flush()
        print("Done in %0.3fs." % (time() - t0))
        return similarity_matrix
                    
mfcc_feat = compute_mfcc()
similarity_matrix = compute_similarity_matrix(mfcc_feat)


print("Normalizing similarity matrix...")
t0 = time()
max = similarity_matrix.max()
min = similarity_matrix.min()
similarity_matrix -= min
similarity_matrix /= (max-min)
print("Done in %0.3fs." % (time() - t0))

print("Visualizing similarity matrix...")
t0 = time()
figure = plt.figure()
plt.title(audio_file)
plt.imshow(similarity_matrix, cmap="gray",interpolation='none',origin='lower')
#plt.show()
figure.savefig(audio_file[:-4]+"_w_"+str(w)+".png")
print("Done in %0.3fs." % (time() - t0))
