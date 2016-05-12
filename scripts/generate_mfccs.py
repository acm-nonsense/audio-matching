from python_speech_features.features.base import mfcc
from python_speech_features.features.base import logfbank

import scipy.io.wavfile as wav
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import sys
import os
from time import time
import warnings
from multiprocessing import Pool,Value

# actually, we're not computing the cross-dot=products, just the diagonals (sampling_window_length sums)... so maybe we can try doing that 
# Should take an audio file, compute the mfcc (which is basically a feature vector), then computes the similarity matrix with square dimensions (dim(mfcc)/sampling_window_length)-squared by averaging over every cross=dot=product for a window of size sampling_window_length
# thus, there are sampling_window_length*sampling_window_length sums


sampling_window_length = 10
file_head_length = 0 # in seconds
warnings.filterwarnings("ignore", category=wav.WavFileWarning)

def compute_mfcc(audio_file):
    #print("\tComputing MFCCs...")
    #t0 = time()
    (rate,sig) = wav.read(audio_file)
    if len(sig.shape) > 1:
        sig = np.mean(sig,axis=1) # Then we have stero channels and we need to flatten them
    if file_head_length != 0: # just take the first file_head_length seconds of the audio file
        sig = sig[0:rate*file_head_length] # delete this after testing
    mfcc_feat = mfcc(sig,rate)
    #fbank_feat = logfbank(sig,rate) # potentially look at this later?
    #print("\tDone in %0.3fs." % (time() - t0))
    return mfcc_feat

def standardize_mfcc_features(mfcc_features):
    #print("\tStandardizing MFCC Features...")
    #t0 = time()
    feature_means = mean(mfcc_features)
    feature_stds = std(mfcc_features)
    standardized_features = (mfcc_features-feature_means)/feature_stds
    #print("\tDone in %0.3fs." % (time() - t0))
    return standardized_features

'''
Inputs: audio_file_path - path to the audio file
standardize_flag - True (default) to standardize the mfccs

Outputs the generated mfcc .npz file to the output path 
'''
def generate_mfcc(item):
    global index
    global total_files

    input_file_path = item["input_file_path"]
    output_file_path = item["output_file_path"]
    standardize_flag = item["flag"]
    mfcc = compute_mfcc(input_file_path)
    if standardize_flag:
        mfcc = standardize_mfcc_features(mfcc)
    index.value += 1
    np.savez(output_file_path, mfcc)
    print "\t{0:2.0f}%\b\b\b\b\b".format(100*index.value/total_files),
    sys.stdout.flush()

index = Value('i',0)

def generate_all_mfccs(input_dir_path, output_dir_path,standardize_flag):
    print("\tGenerating MFCCs...")
    t0 = time()
    global total_files
    total_files = float(len( os.listdir(input_dir_path)))
    pool = Pool(6)
    items = []
    for target_file in os.listdir(input_dir_path):
        if target_file[-4:] == ".wav":
            input_file_path = os.path.join(input_dir_path,target_file)
            output_file_path = os.path.join(output_dir_path,"npz",target_file[:-4]+".mfcc.npz")
            items.append({
                "input_file_path": input_file_path,
                "output_file_path": output_file_path,
                "flag": standardize_flag
            })
    pool.map(generate_mfcc,items)
    print("\tDone in %0.3fs." % (time() - t0))

'''
Inputs: source_path
output_path
standardize_flag - t/f for true/false

Writes .npz files to output_path/npz/mfcc/*, completely copying the directory layout
'''
if __name__ == "__main__":
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("usage: python scripts/{} source_path output_path standardize_flag".format(os.path.basename(__file__)))
        sys.exit(1)

    source_path = sys.argv[1]
    output_path = sys.argv[2]
    if not os.path.exists(os.path.join(output_path,"npz")):
        os.mkdir(os.path.join(output_path,"npz"))
    if not os.path.exists(os.path.join(output_path,"img")):
        os.mkdir(os.path.join(output_path,"img"))
    if len(sys.argv) == 4:
        standardize_flag = (sys.argv[3] == 't')
        generate_all_mfccs(source_path,output_path,standardize_flag)
    else:
        generate_all_mfccs(source_path,output_path)
