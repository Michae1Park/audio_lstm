import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation

from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import librosa

# convert an array of values into a dataset matrix
def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)

# fix random seed for reproducibility
np.random.seed(7)

# load the dataset
# dataframe = read_csv('./data/international-airline-passengers.csv', usecols=[1], engine='python', skipfooter=3)
# dataset = dataframe.values
# dataset = dataset.astype('float32')

t = np.linspace(0.0, np.pi*2.0, 100)
n = 1000
X = []
X1 = []
X2 = []
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
    # x1 = np.cos(t)
    # x2 = np.sin(t)
    X1.append(x1)
    X2.append(x2)

X1 = np.array(X1)
X2 = np.array(X2)
print X1.shape, X2.shape

X1c = X1[0]
X2c = X2[0]
for i in range(1, len(X1)):
    X1c = np.concatenate((X1c, X1[i]))#, axis=0)
    X2c = np.concatenate((X2c, X2[i]))#, axis=0)
print X1c.shape, X2c.shape

X = np.vstack((X1c, X2c))
print 'xshape'
print X.shape

#Plot - Sanity Check
t = np.linspace(0.0, np.pi*2.0, 100*1000)
font = {'size'   : 9}
matplotlib.rc('font', **font)
f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
f0.suptitle("ARtag & Audio combined Prediction", fontsize=12)
ax01 = subplot2grid((4, 2), (0, 0))
ax02 = subplot2grid((4, 2), (1, 0))
ax03 = subplot2grid((4, 2), (2, 0))
ax04 = subplot2grid((4, 2), (3, 0))
ax05 = subplot2grid((4, 2), (0, 1))
ax06 = subplot2grid((4, 2), (1, 1))
ax07 = subplot2grid((4, 2), (2, 1))
ax08 = subplot2grid((4, 2), (3, 1))
#X[0]=cos, X[1]=sin
ax01.grid(True)
ax01.plot(t, X[0])
ax02.grid(True)
ax02.plot(t, X[1])
plt.show()

#Pack into correct dataset shape
print 'packing'
print X.shape
dataset = X
dataset = dataset.astype('float32')
print 'dataset shape'
print dataset.shape

# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset1 = scaler.fit_transform(dataset[0])
dataset2 = scaler.fit_transform(dataset[1])

print 'scaler'
print dataset1.shape, dataset2.shape

# split into train and test sets
train_size = int(len(dataset) * 0.67)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 4
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# reshape input to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], 1))
testX = np.reshape(testX, (testX.shape[0], testX.shape[1], 1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(look_back, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
model.save_weights('./models/my_model_weights_sincos.h5')

# make predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

# shift train predictions for plotting
trainPredictPlot = np.empty_like(dataset)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(trainPredict)+look_back, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(dataset)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(look_back*2)+1:len(dataset)-1, :] = testPredict

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.show()
