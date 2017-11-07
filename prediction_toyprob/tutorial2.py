import numpy as np
import matplotlib.pyplot as plt
from pandas import read_csv
import math

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

y, sr = librosa.load('./data/data13.wav', sr=8000)
dataset = y
print dataset.shape

tmp2 = []
for data in dataset:
	tmp = []
	tmp.append(data)
	tmp2.append(tmp)

dataset = np.array(tmp2)
dataset = dataset.astype('float32')
print dataset.shape


# normalize the dataset
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)

# split into train and test sets
train_size = int(len(dataset) * 0.1)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size,:], dataset[train_size:len(dataset),:]

# reshape into X=t and Y=t+1
look_back = 3
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
# model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)
# model.save_weights('./models/my_model_weights.h5')
model.load_weights('./models/my_model_weights_onemicrowave_lookback3.h5')

#--------Insert Sequence Prediction part -------
print testX.shape

# make predictions - Original
trainPredict = model.predict(trainX)
testPredict = model.predict(testX) 	
print testPredict.shape

# make predictions - Sequence by Sequence
seq = []
flag = True
for x in testX:
	x = np.expand_dims(x, axis=0)
	testPredict = model.predict(x)
	if flag:
		seq = testPredict
		flag = False
	else:
		seq = np.vstack((seq,testPredict))
testPredict = np.array(seq)
print testPredict.shape

# make predictions - Prediction Based
# seq = []
# flag = 0
# x_f = [] #contains the first three elements in the first window
# x_n = []
# t_p = []
# for x in testX:
# 	if flag == 0:
# 		x_f = x
# 		x_f = np.expand_dims(x_f, axis=0)
# 		testPredict = model.predict(x_f)
# 		x_n = testPredict
# 		seq = testPredict	
# 		flag = 1
# 	elif flag == 1:
# 		x_f = x_f[:,1:3,:]
# 		x_n = np.expand_dims(x_n, axis=1)
# 		x_n = np.concatenate((x_f, x_n), axis=1)
# 		testPredict = model.predict(x_n)
# 		t_p = testPredict
# 		seq = np.vstack((seq, testPredict))
# 		flag = 2
# 	elif flag == 2:
# 		x_f = x_f[:,2:3,:]
# 		first = np.expand_dims(first, axis=1)
# 		first = np.concatenate((x_f, first), axis=1)
# 		testPredict = model.predict(first)
# 		first = testPredict
# 		seq = np.vstack((seq,testPredict))
# 		flag = 3
# 	elif flag == 3:
# 		x_f = x_f[:,3:3,:]
# 		first = np.expand_dims(first, axis=1)
# 		first = np.concatenate((x_f, first), axis=1)
# 		testPredict = model.predict(first)
# 		first = testPredict
# 		seq = np.vstack((seq,testPredict))
# 		flag = 4
# 	else:
# 		print 'c'

# testPredict = np.array(seq)
# print testPredict.shape

#-----------------------------------------

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

librosa.output.write_wav('./data/data13_predicted', testPredict, 8000)

