import json
from scipy.io.wavfile import read
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pylab
import sys

if len(sys.argv) < 2:
	print("usage: python generate_spectrograms.py audio.wav")
	sys.exit(1)

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
WINDOW_INTERVAL = 1 # In seconds
WINDOW_LENGTH_SAMPLES = WINDOW_LENGTH_SECONDS * SAMPLE_RATE

channel_windows = []
for i in range(LENGTH_SECONDS/WINDOW_INTERVAL):
	channel_windows.append(channel[SAMPLE_RATE*i:SAMPLE_RATE*(i+WINDOW_LENGTH_SECONDS)])

print("Created channel windows.")

print 'Generating spectrograms: ',


SPECTRUM_WIDTH = 1721
SPECTRUM_HEIGHT = 129
spectrograms = np.ndarray((0,SPECTRUM_HEIGHT,SPECTRUM_WIDTH))
for i,window in enumerate(channel_windows):
	figure = plt.figure()
	spectrum, freqs, bins, plot = pylab.specgram(
		window,
		NFFT=256, 
		Fs=44100, 
		detrend=pylab.detrend_none,
		window=pylab.window_hanning,
		noverlap=int(256 * 0.5))
	if i == 1:
		bins_static = bins
		freqs_static = freqs
	# plt.pcolormesh(bins, freqs, 10*np.log10(spectrum))
	figure.savefig('../spectrograms/anb-{}.png'.format(i))
	full_dim_spectrum = np.ndarray((1,SPECTRUM_HEIGHT,SPECTRUM_WIDTH))
	if np.shape(spectrograms)[1:3] == np.shape(spectrum)[0:2]:
		full_dim_spectrum[0] = spectrum
		spectrograms = np.concatenate((spectrograms,full_dim_spectrum))
	plt.close()
	print "{0:2.0f}%\b\b\b\b".format(100*float(i)/len(channel_windows)),
	sys.stdout.flush()
specs = open('{}.specs.npz'.format(sys.argv[1]), 'w')
bins = open('{}.bins.npz'.format(sys.argv[1]), 'w')
freqs = open('{}.freqs.npz'.format(sys.argv[1]), 'w')
np.save(specs,spectrograms)
np.save(bins,bins_static)
np.save(freqs,freqs_static)
print 'Done.'

print(bins_static.shape)
print(freqs_static.shape)
print(spectrograms[0].shape)
plt.pcolormesh(bins_static, freqs_static, 10*np.log10(spectrograms[0]))