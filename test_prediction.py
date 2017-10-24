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

from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

#Configurations
SOUND_FOLDER = './sounds/'
EPOCHS = 10
FEATURE_COUNT = 3
N_PRE = 10
N_POST = 1

#Audio Configuration
AUDIO_FILENAME = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
N_MFCC = 2

def create_model(steps_before, steps_after, feature_count):
    DROPOUT = 0.5
    LAYERS = 1
    hidden_neurons = 300

    if LAYERS == 1:
        hidden_neurons = feature_count

    model = Sequential()
    model.add(LSTM(input_dim=feature_count, output_dim=hidden_neurons, return_sequences=False))
    #model.add(LSTM(output_dim=hidden_neurons, return_sequences=True))
    #model.add(TimeDistributed(Dense(feature_count)))
    model.add(Activation('linear'))  
    model.compile(loss='mean_squared_error', optimizer='rmsprop', metrics=['accuracy'])  
    return model

def train_model(model, dataX, dataY, epoch_count):
    #history = model.fit(dataX, dataY, batch_size=3, nb_epoch=epoch_count, validation_split=0.05)

    csv_logger = CSVLogger('training_audio.log')
    escb = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint("models/audio-{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='val_loss', save_best_only=True, verbose=1) #, period=2)

    model.fit(dataX, dataY, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
              validation_split=0.3, nb_epoch=500, callbacks=[csv_logger, escb, checkpoint])

    #matplotlib inline
    print "Training history"
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.plot(model.history.history['loss'])
    ax1.set_title('loss')
    ax2 = fig.add_subplot(1, 2, 2)
    plt.plot(model.history.history['val_loss'])
    ax2.set_title('validation loss')

def normalize(y):
    # normalize - for feeding into LSTM
    min_y = np.min(y)
    max_y = np.max(y)
    y = (y - min_y) / (max_y - min_y)
    print y.dtype, min_y, max_y
    return y

def scale_back():
    # scale back 
    seq = seq * (max_y - min_y) + min_y

def image_to_tensor():
    #image_filename = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
    samples = []

    for image in range(3):
        i = np.arange(92)
        i = i[1:] #1-91
        i = i.reshape((1,91))
        samples.append(i)
    samples = np.array(samples, dtype=float)
    
    #print samples
    print samples.shape
    return samples

def audio_to_tensor():
    #these data have to be in same length
    # audio_filename = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']
    audio_filename = AUDIO_FILENAME
    samples = []
    
    for audio_file in audio_filename:
        y, sr = librosa.load(SOUND_FOLDER + audio_file, mono=True)
        #print y.shape

        mfccs = librosa.feature.mfcc(y, n_mfcc=N_MFCC) #default hop_length=512
        #print mfccs
        #print mfccs.shape
        samples.append(mfccs)    

    samples = np.array(samples, dtype=float)

    #print samples
    print samples.shape        
    return samples

def create_data(n_pre, n_post):
    audio_data = audio_to_tensor()  #returns normalized data packed in correct dim
    image_data = image_to_tensor()  #returns normalized data packed in correct dim

    #Combine audio and image data
    stacked_data = np.concatenate((audio_data, image_data), axis=1)
    print stacked_data.shape
    stacked_data = np.rollaxis(stacked_data, 2, 1) # (array, axis, start=0)
    print stacked_data.shape

    #create training set (X Y pair)
    dataX = stacked_data
    dataY = stacked_data[:,1:,:]
    print dataY.shape
    
    zero = np.zeros((3,3), dtype=np.float64)

    np.concatenate((dataY,zero[:,None,:]),axis=1)
    # ax = 1
    # np.concatenate((np.expand_dims(dataY[:,0,:],ax), dataY), ax) 
    # np.append(dataY, np.atleast_3d(zero), axis=1)
    print dataY.shapes

    #Create input output pair
    # dX, dY = [], []
    # for i in range(stacked_data.shape[2]-n_post): #i starts from 0
    #     dX.append(stacked_data[i:i+n_pre])
    #     dY.append(stacked_data[i+n_pre:i+n_pre+n_post])
    # dataX = np.array(dX)
    # dataY = np.array(dY)

    #final data should have (batch_size=num_samples, time_step, num_features) (eg)(3,10,3)
    # (eg) [dataX: (3,10,3), dataY:(3,1,3)] x 90(total time series)
    return dataX, dataY

def test_prediction():
    os.environ["KERAS_BACKEND"] = "tensorflow"
    
    # have Three options
    # 1) one to one (eg) n_pre=1, n_post=1
    # 2) sequence to one (eg) n_pre=10, n_post=1
    # 3) sequence to sequence (eg) n_pre=10, n_post=10
    n_pre = N_PRE
    n_post = N_POST

    print('creating dataset...')
    dataX, dataY = create_data(n_pre, n_post)
    print dataX.shape, dataY.shape

    # create and fit the LSTM network
    print('creating model...')
    model = create_model(n_pre, n_post, FEATURE_COUNT)

    #normalize data
    dataX = normalize(dataX)
    dataY = normalize(dataY)

    #Train LSTM
    print('training model...')
    train_model(model, dataX, dataY, EPOCHS)
    
    # now test
    t = np.arange(15.0, 19.0, 0.02)
    sinus = np.sin(2 * np.pi * t)
    sinus = sinus.reshape((sinus.shape[0], 1))
    
    dX, dY = [], []
    for i in range(len(sinus)-n_pre-n_post):
        dX.append(sinus[i:i+n_pre])
        dY.append(sinus[i+n_pre:i+n_pre+n_post])
    dataX = np.array(dX)
    dataY = np.array(dY)
    
    predict = model.predict(dataX)
    
    # now plot
    nan_array = np.empty((n_pre - 1))
    nan_array.fill(np.nan)
    nan_array2 = np.empty(n_post)
    nan_array2.fill(np.nan)
    ind = np.arange(n_pre + n_post)

    fig, ax = plt.subplots()
    for i in range(0, 50, 50):

        forecasts = np.concatenate((nan_array, dataX[i, -1:, 0], predict[i, :, 0]))
        ground_truth = np.concatenate((nan_array, dataX[i, -1:, 0], dataY[i, :, 0]))
        network_input = np.concatenate((dataX[i, :, 0], nan_array2))
     
        ax.plot(ind, network_input, 'b-x', label='Network input')
        ax.plot(ind, forecasts, 'r-x', label='Many to many model forecast')
        ax.plot(ind, ground_truth, 'g-x', label = 'Ground truth')
        
        plt.xlabel('t')
        plt.ylabel('sin(t)')
        plt.title('Sinus Many to Many Forecast')
        plt.legend(loc='best')
        plt.savefig('test_sinus/plot_mtm_triple_' + str(i) + '.png')
        plt.cla()

def main():
    test_prediction()
    return 1

if __name__ == "__main__":
    sys.exit(main())

