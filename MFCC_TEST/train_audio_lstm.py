# All public gists https://gist.github.com/rwldrn
# Copyright 2017, Nao Tokui
# MIT License, https://gist.github.com/naotokui/12df40fa0ea315de53391ddc3e9dc0b9

import seaborn
import librosa
import numpy as np

from IPython.display import Audio
import matplotlib.pyplot as plt
import os, copy, sys
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.layers import Input
from keras.optimizers import RMSprop
import tensorflow as tf
from tqdm import tqdm
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from IPython.display import display

# # load array to audio buffer and play!!
# def sample(preds, temperature=1.0, min_value=0, max_value=1):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds)
#     probas = np.random.multinomial(1, preds, 1)
#     v = np.argmax(probas)/float(probas.shape[1])
#     return v * (max_value - min_value) + min_value

# 1. Preprocess data pack into (time, mfcc=2, num_samples)
# 2. Define LSTM network
# 3. Train LSTM and check convergence
# 4. Preidct
# 5. Reconstruct - denormalize and reverse mfcc?

def define_network(maxlen, nb_output, latent_dim):
    inputs = Input(shape=(maxlen, nb_output))
    x = LSTM(latent_dim, return_sequences=True)(inputs)
    # x = Dropout(0.2)(x)         #prevents overfitting - fraction of input to drop
    # x = LSTM(latent_dim)(x)
    # x = Dropout(0.2)(x)
    output = Dense(nb_output, activation='softmax')(x)
    model = Model(inputs, output)

    #optimizer = Adam(lr=0.005)
    optimizer = RMSprop(lr=0.01) 
    model.compile(loss='mean_squared_error', optimizer=optimizer)
    #model.compile(loss='categorical_crossentropy', optimizer=optimizer)
    return model

def convert_to_tensor(maxlen, nb_output, latent_dim):
    #these data have to be in same length
    audio_filename = ['data1crop4.wav', 'data2crop4.wav', 'data5crop4.wav']

    for audio_file in audio_filename:
        y, sr = librosa.load(audio_file, mono=True)
        print y.shape

        #first divide into time chunks then convert to mfcc
        #Without Converting to Binary
        mfccs = librosa.feature.mfcc(y, n_mfcc=2) #default hop_length=512
        # print mfccs
        # print mfccs.shape
        
        # normalize - for feeding into LSTM
        min_mfcc = np.min(mfccs)
        max_mfcc = np.max(mfccs)
        mfccs = (mfccs - min_mfcc) / (max_mfcc - min_mfcc)
        print mfccs.dtype, min_mfcc, max_mfcc

        print mfccs
        print mfccs.shape

        # samples = np.array(samples, dtype=float)
        # next_sample = np.array(next_sample, dtype=float)
        # print samples.shape, next_sample.shape        

    return samples, next_sample

def main():
    # Configurations
    # Build a model
    os.environ["KERAS_BACKEND"] = "tensorflow"
    # so try to estimate next sample afte given (maxlen) samples
    maxlen     = 256 # 256/44100 = 0.012s AKA framesize
    #nb_output = 256  # resolution - 8bit encoding - output of hidden layers?
    nb_output = 2 # 2-dim mfcc data
    #latent_dim = 128 #dimensionality of the output space
    latent_dim = 2048 #hidden dimension I think

    #1. Preprocess Data
    samples, next_sample = convert_to_tensor(maxlen, nb_output, latent_dim)

    #2. Define network
    model  = define_network(maxlen, nb_output, latent_dim)

    #3. Train network
    csv_logger = CSVLogger('training_audio.log')
    escb = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
    checkpoint = ModelCheckpoint("models/audio-{epoch:02d}-{val_loss:.2f}.hdf5", 
        monitor='val_loss', save_best_only=True, verbose=1) #, period=2)

    model.fit(samples, next_sample, shuffle=True, batch_size=256, verbose=1, #initial_epoch=50,
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
    
    ###### BELOW IS REDUNDANT IN TRAINING PHASE ########
    #Below just for plotting train history     
    seqA = []
    for start in range(5000,220000,10000):
        seq = y[start: maxlen]  
        seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
        for i,s in enumerate(seq):
            sample_ = int(s * (nb_output - 1)) # 0-255
            seq_matrix[i, sample_] = True

        for i in tqdm(range(5000)):
            z = model.predict(seq_matrix.reshape((1,maxlen,nb_output)))
            s = sample(z[0], 1.0)
            seq = np.append(seq, s)

            sample_ = int(s * (nb_output - 1))    
            seq_vec = np.zeros(nb_output, dtype=bool)
            seq_vec[sample_] = True

            seq_matrix = np.vstack((seq_matrix, seq_vec))  # added generated note info 
            seq_matrix = seq_matrix[1:]
            
        # scale back 
        seq = seq * (max_y - min_y) + min_y

        # plot
        plt.figure(figsize=(30,5))
        plt.plot(seq.transpose())
        plt.show()
        
        display(Audio(seq, rate=sr))
        print seq
        seqA.append(seq)
        #join seq data
    
    seqA2 = np.hstack(seqA)
    librosa.output.write_wav('data1crop4_predictwav', seqA2, sr)


if __name__ == '__main__':
    main()


