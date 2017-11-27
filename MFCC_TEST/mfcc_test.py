# Start up ROS pieces.
PKG = 'my_package'
# import roslib; roslib.load_manifest(PKG)
import os, copy, sys

## from hrl_msgs.msg import FloatArray
## from std_msgs.msg import Float64
from hrl_anomaly_detection.msg import audio

# util
import numpy as np
import math
import pyaudio
import struct
import array
try:
    from features import mfcc
except:
    from python_speech_features import mfcc
from scipy import signal, fftpack, conj, stats

import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
# from matplotlib import cm

import librosa
import librosa.display

N_MFCC = 3

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def audio_creator(): 
    WAVE_OUTPUT_FILENAME = "data['1'].wav" #n_mfcc=24
    #WAVE_OUTPUT_FILENAME = "AudioBookCrop.wav" #n_mfcc=12
    #WAVE_OUTPUT_FILENAME = "data5crop4.wav"  #n_mfcc=2
    
    y, sr = librosa.load('./sounds/' + WAVE_OUTPUT_FILENAME)
    print len(y)

    #calculate mfccs
    #Y = librosa.stft(y)
    #print Y.shape

    #####################################
    # Original #
    # if unspecified win_length=n_fft
    # hop_length=win_length/4
    # n_fft = fft window size
    # 
    mfccs = librosa.feature.mfcc(y=y, sr=44100, hop_length=1024, n_fft=4096, n_mfcc=N_MFCC)# default hop_length=512, hop_length=int(0.01*sr))
    print 'mfccs shape'
    print mfccs.shape
    ############################

    #build reconstruction mappings
    n_mfcc = mfccs.shape[0]
    n_mel = 128 #just using n_mel = hop_length/4 based on the default values given in code
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 4096
    mel_basis = librosa.filters.mel(sr, n_fft, n_mels=n_mel)

    #Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    #Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    print bin_scaling[:, np.newaxis].shape
    print '------'
    print mel_basis.T.shape
    print np.dot(dctm.T, mfccs).shape
    print np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs))).shape
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
    #Impose reconstructed magnitude on white noise STFT.
    excitation = np.random.randn(y.shape[0])
    print y.shape[0]
    E = librosa.stft(excitation, n_fft=n_fft)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    #print recon
    #print recon.shape

    wav.write('./sounds/' + WAVE_OUTPUT_FILENAME +'FromMFCC', sr, recon)

def main():
    audio_creator()

if __name__ == '__main__':
    main()


