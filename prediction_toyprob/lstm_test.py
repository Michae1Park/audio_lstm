import numpy as np
import sys, os
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation

#####################################################
#Objective of this is to learn how to tune my model & params
#####################################################

#Generate Sin Cos wave for 2 dimensional data
#Augment data at least 1000 samples - 1)slight change in freq, 2)phase shift, 3)magnitude
t = np.linspace(0.0, np.pi*2.0, 100)
n = 1000

X = []
for i in xrange(n):
    if i<1:
        x1 = np.cos(t) + np.random.normal(-0.3, 0.3, np.shape(t) ) #random.normal(mu, sig, )
        x2 = np.sin(t) + np.random.normal(-0.3, 0.3, np.shape(t) )
    elif i<100:
        x1 = np.cos(t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = np.sin(t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    elif i<200:
        x1 = np.cos(t) + np.random.normal(-0.1, 0.1, np.shape(t) )
        x2 = np.sin(t) + np.random.normal(-0.1, 0.1, np.shape(t) )        
    elif i<300:
        x1 = np.cos(1.1*t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = np.sin(1.1*t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    elif i<400:
        x1 = np.cos(0.8*t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = np.sin(0.8*t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    elif i<500:
        x1 = np.cos(1.1*t) + np.random.normal(-0.3, 0.3, np.shape(t) )
        x2 = np.sin(1.1*t) + np.random.normal(-0.3, 0.3, np.shape(t) )        
    elif i<600:
        x1 = 1.2*np.cos(t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = 1.2*np.sin(t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    elif i<700:
        x1 = 1.1*np.cos(0.9*t) + np.random.normal(-0.1, 0.1, np.shape(t) )
        x2 = 1.1*np.sin(0.9*t) + np.random.normal(-0.1, 0.1, np.shape(t) )        
    elif i<800:
        x1 = 0.8*np.cos(0.9*t) + np.random.normal(-0.3, 0.3, np.shape(t) )
        x2 = 0.8*np.sin(0.9*t) + np.random.normal(-0.3, 0.3, np.shape(t) )        
    elif i<900:
        x1 = np.cos(0.9*t) + np.random.normal(-0.3, 0.3, np.shape(t) )
        x2 = np.sin(0.9*t) + np.random.normal(-0.3, 0.3, np.shape(t) )        
    elif i<1000:
        x1 = 1.1*np.cos(0.9*t) + np.random.normal(-0.2, 0.2, np.shape(t) )
        x2 = 1.1*np.sin(0.9*t) + np.random.normal(-0.2, 0.2, np.shape(t) )        
    X.append( np.vstack([ x1.reshape(1,len(t)), x2.reshape(1,len(t)) ]) )
X = np.swapaxes(X, 0,1)

# #Plot 
# font = {'size'   : 9}
# matplotlib.rc('font', **font)
# f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
# f0.suptitle("ARtag & Audio combined Prediction", fontsize=12)
# ax01 = subplot2grid((4, 2), (0, 0))
# ax02 = subplot2grid((4, 2), (1, 0))
# ax03 = subplot2grid((4, 2), (2, 0))
# ax04 = subplot2grid((4, 2), (3, 0))
# ax05 = subplot2grid((4, 2), (0, 1))
# ax06 = subplot2grid((4, 2), (1, 1))
# ax07 = subplot2grid((4, 2), (2, 1))
# ax08 = subplot2grid((4, 2), (3, 1))
# #X[0]=cos, X[1]=sin
# ax01.grid(True)
# ax01.plot(t, X[0][101])
# ax02.grid(True)
# ax02.plot(t, X[0][901])
# ax05.grid(True)
# ax05.plot(t, X[1][0])
# ax06.grid(True)
# ax06.plot(t, X[1][1])
# plt.show()

#Generate Data like I do - Window method
# print 'a'
# print X.shape
# window_size = 5

# dX, dY = [], []
# for i in range(n - window_size + 1):
#     dX.append(X[i : i+window_size])
#     dY.append(X[i+window_size : i+window_size+1][0]) 

# #window over entire dataset - mfcc
# mfccs = np.rollaxis(mfccs, 1, 0)
# aX, aY = self.construct_dataset(mfccs)
# audio_dataX.append(aX)
# audio_dataY.append(aY)
# #window over entire dataset - xyz
# iX, iY = self.construct_dataset(relative_position_intp)
# image_dataX.append(iX)
# image_dataY.append(iY)

# #Below here should be outside the loop
# audio_dataX = np.array(audio_dataX)
# audio_dataY = np.array(audio_dataY)
# image_dataX = np.array(image_dataX)
# image_dataY = np.array(image_dataY)


# #concatenate for number of experiment samples
# audio_dataX2 = audio_dataX[0]
# audio_dataY2 = audio_dataY[0]
# image_dataX2 = image_dataX[0]
# image_dataY2 = image_dataY[0]
# for i in range(1, audio_dataX.shape[0]):
#     audio_dataX2 = np.concatenate((audio_dataX2, audio_dataX[i]), axis=0)
#     audio_dataY2 = np.concatenate((audio_dataY2, audio_dataY[i]), axis=0)
#     image_dataX2 = np.concatenate((image_dataX2, image_dataX[i]), axis=0)
#     image_dataY2 = np.concatenate((image_dataY2, image_dataY[i]), axis=0)
# audio_dataX = audio_dataX2
# audio_dataY = audio_dataY2
# image_dataX = image_dataX2
# image_dataY = image_dataY2
    

#Use LSTM to train/predict based on predict, try as much tuning as possible



#Plot final result





#Optional
#See what I get using SVM


