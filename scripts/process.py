from scipy.io.wavfile import read
from scipy import signal
from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
import pylab

# Read file to get samplerate and numpy array containing the signal 
(fs, x) = read("../samples/layers.wav")


channels = [
    np.array(x[:, 0]),
    np.array(x[:, 1])
]

# Combine channels to make a mono signal out of stereo
channel =  np.mean(channels, axis=0)

# Generate spectrogram 
## Freqs is the same with different songs, t differs slightly
Pxx, freqs, t, plot = pylab.specgram(
    channel,
    NFFT=2048, 
    Fs=44100, 
    detrend=pylab.detrend_none,
    window=pylab.window_hanning,
    noverlap=int(2048 * 0.5))