#!/usr/bin/python
#
# Copyright (c) 2017, Georgia Tech Research Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * Neither the name of the Georgia Tech Research Corporation nor the
#       names of its contributors may be used to endorse or promote products
#       derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY GEORGIA TECH RESEARCH CORPORATION ''AS IS'' AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL GEORGIA TECH BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
# LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
# OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
# OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
#  \author Michael Park (Healthcare Robotics Lab, Georgia Tech.)

from attrdict import AttrDict
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import librosa
import os, copy, sys
import tensorflow as tf
from tqdm import tqdm
import numpy.matlib
import scipy.io.wavfile as wav

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

#Configurations
SOUND_FOLDER = './sounds/'
EPOCHS = 10
# N_PRE = 10
# N_POST = 1
AUDIO_DATA = 1
IMAGE_DATA = 0
DATA_COMBINE = 0

#Audio Configuration
AUDIO_FILENAME = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
N_MFCC = 2

#LSTM Configuration
NUM_SAMPLES = 3 # N aka number of experiments
NUM_FEATURE = 2 #Dimension in LSTM
NUM_TIME_SAMPLE = 91 #Number of total time samples
WINDOW_SIZE_IN = 5
WINDOW_SIZE_OUT = 1 
LOAD_WEIGHT = 1
NUM_STEP_SHOW = 86

def create_model():
    # DROPOUT = 0.5
    # LAYERS = 1

    # For multiple LSTMS
    # hidden_neurons = 300
    # if LAYERS == 1:
    #     hidden_neurons = feature_count

    model = Sequential()
    model.add(LSTM(output_dim=NUM_FEATURE, input_shape=(WINDOW_SIZE_IN, NUM_FEATURE)))
    #model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    #model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))  

    if LOAD_WEIGHT == 1:
        model.load_weights('/home/mpark/approach2.hdf5')

    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataX, dataY):
    #history = model.fit(dataX, dataY, batch_size=3, nb_epoch=epoch_count, validation_split=0.05)

    csv_logger = CSVLogger('training_audio.log')
    escb = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint("models/audio-{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='val_loss', save_best_only=True, verbose=1) #, period=2)

    model.fit(dataX, dataY, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
              validation_split=0.3, nb_epoch=500, callbacks=[csv_logger, escb, checkpoint])

    #matplotlib inline
    # print "Training history"
    # fig = plt.figure(figsize=(10,4))
    # ax1 = fig.add_subplot(1, 2, 1)
    # plt.plot(model.history.history['loss'])
    # ax1.set_title('loss')
    # ax2 = fig.add_subplot(1, 2, 2)
    # plt.plot(model.history.history['val_loss'])
    # ax2.set_title('validation loss')

def normalize(y):
    # normalize - for feeding into LSTM
    min_y = np.min(y)
    max_y = np.max(y)
    y = (y - min_y) / (max_y - min_y)
    #print y.dtype, min_y, max_y
    return y, min_y, max_y

def scale_back(seq, min_y, max_y):
    # scale back 
    seq = seq * (max_y - min_y) + min_y
    return seq

def image_to_tensor():
    #image_filename = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
    samples = []

    for image in range(NUM_SAMPLES):
        i = np.arange(92)
        i = i[1:] #1-91
        i = i.reshape((1,91))
        samples.append(i)

    samples, min_samples, max_samples = normalize(samples)
    samples = np.array(samples, dtype=float)
    #print samples
    print samples.shape
    return samples

def audio_to_tensor(audio_filename):
    #these data have to be in same length
    # audio_filename = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
    audio_dataX = []
    audio_dataY = []

    for audio_file in audio_filename:
        y, sr = librosa.load(SOUND_FOLDER + audio_file, mono=True)
        mfccs = librosa.feature.mfcc(y, n_mfcc=N_MFCC) #default hop_length=512
        mfccs = np.rollaxis(mfccs, 1, 0)
        dX, dY = [], []
        for i in range(mfccs.shape[0] - WINDOW_SIZE_IN):
                dX.append(mfccs[i:i+WINDOW_SIZE_IN])
                dY.append(mfccs[i+WINDOW_SIZE_IN:i+WINDOW_SIZE_IN+WINDOW_SIZE_OUT][0])
        audio_dataX = np.array(dX)
        audio_dataY = np.array(dY)

    #normalization should be done feature by feature
    audio_dataX, min_audio_dataX, max_audio_dataX = normalize(audio_dataX) 
    audio_dataY, min_audio_dataY, max_audio_dataY = normalize(audio_dataY) 

    audio_dataX = np.array(audio_dataX, dtype=float)
    audio_dataY = np.array(audio_dataY, dtype=float) 
    print audio_dataX.shape, audio_dataY.shape    

    return sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY

def create_data(n_pre, n_post, audio_filename):
    if AUDIO_DATA == 1:
        sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY = audio_to_tensor(audio_filename)  #returns normalized data packed in correct dim
    if IMAGE_DATA == 1:
        image_data = image_to_tensor()  #returns normalized data packed in correct dim

    if DATA_COMBINE == 1:
        print 'using multi modality'
        #Combine audio and image data
        stacked_data = np.concatenate((audio_data, image_data), axis=1)
        print stacked_data.shape
        stacked_data = np.rollaxis(stacked_data, 2, 1) # (array, axis, start=0)
        print stacked_data.shape

        #create training set (X Y pair)
        dataX = stacked_data
        dataY = stacked_data[:,1:,:]
        print dataY.shape    
        dataY = np.pad(dataY, ((0,0), (0,1), (0,0)), mode='constant', constant_values = 0)
        print dataY.shape
    else:
        print 'using single modality'

    #final data should have (batch_size=num_samples, time_step, num_features) (eg)(3,10,3)
    # (eg) [dataX: (3,10,3), dataY:(3,1,3)] x 90(total time series)
    return sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY

def invlogamplitude(S):
#"""librosa.logamplitude is actually 10_log10, so invert that."""
    return 10.0**(S/10.0)

def reconstruct_audio(mfccs, sr, y_shape):
    #build reconstruction mappings
    n_mfcc = mfccs.shape[0]
    n_mel = 128
    dctm = librosa.filters.dct(n_mfcc, n_mel)
    n_fft = 2048
    mel_basis = librosa.filters.mel(sr, n_fft)

    #Empirical scaling of channels to get ~flat amplitude mapping.
    bin_scaling = 1.0/np.maximum(0.0005, np.sum(np.dot(mel_basis.T, mel_basis), axis=0))
    #Reconstruct the approximate STFT squared-magnitude from the MFCCs.
    recon_stft = bin_scaling[:, np.newaxis] * np.dot(mel_basis.T, invlogamplitude(np.dot(dctm.T, mfccs)))
    #Impose reconstructed magnitude on white noise STFT.
    #tot_timeseq = 91
    #y = np.zeros((N_MFCC,tot_timeseq))
    excitation = np.random.randn(y_shape)
    E = librosa.stft(excitation)
    recon = librosa.istft(E/np.abs(E)*np.sqrt(recon_stft))
    #print recon
    #print recon.shape

    wav.write('./sounds/' + 'test_predict' +'FromMFCC', sr, recon)

def test_prediction():
    os.environ["KERAS_BACKEND"] = "tensorflow"
    # have Three options
    # 1) one to one (eg) n_pre=1, n_post=1
    # 2) sequence to one (eg) n_pre=10, n_post=1
    # 3) sequence to sequence (eg) n_pre=10, n_post=10
    # n_pre = N_PRE
    # n_post = N_POST

    # print('creating dataset...')
    # sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY = create_data(WINDOW_SIZE_IN, WINDOW_SIZE_OUT, AUDIO_FILENAME)
    # create and fit the LSTM network
    print('creating model...')
    model = create_model()
    # #Train print
    #LSTM('training model...')
    #train_model(model, audio_dataX, audio_dataY)
    
    # ******************************************************** #
    # Testing Phase - Just comment out the train_model function
    # Prepare Testing Data
    sr, audio_dataX, min_audio_dataX, max_audio_dataX, audio_dataY, min_audio_dataY, max_audio_dataY = create_data(WINDOW_SIZE_IN, WINDOW_SIZE_OUT, ['data5crop4.wav'])
    #datain = audio_dataX[0:NUM_STEP_SHOW,:,:]
    datain = audio_dataX
    print 'shape?'
    print datain.shape
    y_shape = 512*(NUM_STEP_SHOW-1)

    audio_predict = model.predict(datain)
    audio_predict = np.rollaxis(audio_predict, 1, 0) # (array, axis, start=0)
    print 'predicted test data'
    print audio_predict.shape
    # y, sr = librosa.load(SOUND_FOLDER + 'data5crop4.wav', mono=True)
    # mfccs = librosa.feature.mfcc(y, n_mfcc=N_MFCC) #default hop_length=512
    print audio_predict
    audio_predict = scale_back(audio_predict, min_audio_dataX, max_audio_dataX)
    print audio_predict
    reconstruct_audio(audio_predict, sr, y_shape)

    # *********** Plotting Not Necessary ********* #
    # now plot
    # nan_array = np.empty((n_pre - 1))
    # nan_array.fill(np.nan)
    # nan_array2 = np.empty(n_post)
    # nan_array2.fill(np.nan)
    # ind = np.arange(n_pre + n_post)

    # fig, ax = plt.subplots()
    # for i in range(0, 50, 50):

    #     forecasts = np.concatenate((nan_array, dataX[i, -1:, 0], predict[i, :, 0]))
    #     ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0], dataY[i, :, 0]))
    #     network_input = np.concatenate((dataX[i, :, 0], nan_array2))
     
    #     ax.plot(ind, network_input, 'b-x', label='Network input')
    #     ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
    #     ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
        
    #     plt.xlabel('t')
    #     plt.ylabel('sin(t)')
    #     plt.title('Sinus Many to Many Forecast')
    #     plt.legend(loc='best')
    #     plt.savefig('test_sinus/plot_mtm_triple_' + str(i) + '.png')
    #     plt.cla()

def main():
    test_prediction()
    return 1

if __name__ == "__main__":
    sys.exit(main())

