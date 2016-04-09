from scipy.io.wavfile import read
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pylab

# Read file to get samplerate and numpy array containing the signal 
(fs, x) = read("../samples/all_names_bond.wav")


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
WINDOW_INTERVAL = 1 # In seconds
WINDOW_LENGTH_SAMPLES = WINDOW_LENGTH_SECONDS * SAMPLE_RATE

channel_windows = []
for i in range(LENGTH_SECONDS/WINDOW_INTERVAL):
	channel_windows.append(channel[SAMPLE_RATE*i:SAMPLE_RATE*(i+WINDOW_LENGTH_SECONDS)])

print(channel_windows)

for i,window in enumerate(channel_windows):
	figure = plt.figure()
	Pxx, freqs, t, plot = pylab.specgram(
	    window,
	    NFFT=2048, 
	    Fs=44100, 
	    detrend=pylab.detrend_none,
	    window=pylab.window_hanning,
	    noverlap=int(2048 * 0.5))
	figure.savefig('../spectrograms/anb-{}.png'.format(i))
	plt.close()
	print(float(i)/len(channel_windows))