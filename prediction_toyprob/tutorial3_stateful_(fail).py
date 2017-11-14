from pandas import DataFrame
from pandas import Series
from pandas import concat
from pandas import read_csv
from pandas import datetime
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt
# from matplotlib import pyplot
from numpy import array
import numpy as np

import matplotlib.pyplot as plt
# import math
# from matplotlib.pylab import *
# from mpl_toolkits.axes_grid1 import host_subplot
# import matplotlib.animation as animation

#program config
SINCOS = True
 
# date-time parsing function for loading the dataset
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
# convert time series into supervised learning problem
def series_to_supervised(data, n_in, n_out, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
 
# create a differenced series
def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return Series(diff)
 
# transform series into train and test sets for supervised learning
def prepare_data(series, n_test, n_lag, n_seq):
	# extract raw values
	raw_values = series.values
	if SINCOS:
		#packing into diff shape except that it is not diffed
		diff = list()
		for i in range(1, len(raw_values)):
			value = raw_values[i]
			diff.append(value)
		diff_values = Series(diff)
	else:
		# transform data to be stationary
		diff_series = difference(raw_values, 1)
		diff_values = diff_series.diff_values
		values = diff_values.reshape(len(diff_values), 1)
	# rescale values to -1, 1
	scaler = MinMaxScaler(feature_range=(-1, 1))
	scaled_values = scaler.fit_transform(diff_values)
	scaled_values = scaled_values.reshape(len(scaled_values), 1)
	# transform into supervised learning problem X, y
	supervised = series_to_supervised(scaled_values, n_lag, n_seq)
	supervised_values = supervised.values
	# split into train and test sets
	train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
	return scaler, train, test
 
# fit an LSTM network to training data
def fit_lstm(train, n_lag, n_seq, n_batch, nb_epoch, n_neurons):
	# reshape training into [samples, timesteps, features]
	if not SINCOS:
		X, y = train[:, 0:n_lag], train[:, n_lag:]
		X = X.reshape(X.shape[0], 1, X.shape[1])
	# design network
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(n_batch, X.shape[1], X.shape[2]), stateful=True))
	model.add(Dense(y.shape[1]))
	model.compile(loss='mean_squared_error', optimizer='adam')
	# fit network
	for i in range(nb_epoch):
		model.fit(X, y, nb_epoch=1, batch_size=n_batch, verbose=1, shuffle=False, show_accuracy=True)
		model.reset_states()
	model.save_weights('./models/stateful.h5')
	# model.load_weights('./models/my_model_weights_onemicrowave_lookback3.h5')
	return model

def plot():
	print 'plotting'

# make one forecast with an LSTM,
def forecast_lstm(model, X, n_batch):
	# reshape input pattern to [samples, timesteps, features]
	X = X.reshape(1, 1, len(X))
	# make forecast
	forecast = model.predict(X, batch_size=n_batch)
	# convert to array
	return [x for x in forecast[0, :]]
 
# evaluate the persistence model
def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
	forecasts = list()
	for i in range(len(test)):
		X, y = test[i, 0:n_lag], test[i, n_lag:]
		# make forecast
		forecast = forecast_lstm(model, X, n_batch)
		# store the forecast
		forecasts.append(forecast)
	return forecasts
 
# invert differenced forecast
def inverse_difference(last_ob, forecast):
	# invert first forecast
	inverted = list()
	inverted.append(forecast[0] + last_ob)
	# propagate difference forecast using inverted first value
	for i in range(1, len(forecast)):
		inverted.append(forecast[i] + inverted[i-1])
	return inverted
 
# inverse data transform on forecasts
def inverse_transform(series, forecasts, scaler, n_test):
	inverted = list()
	for i in range(len(forecasts)):
		# create array from forecast
		forecast = array(forecasts[i])
		forecast = forecast.reshape(1, len(forecast))
		# invert scaling
		inv_scale = scaler.inverse_transform(forecast)
		inv_scale = inv_scale[0, :]
		# invert differencing
		index = len(series) - n_test + i - 1
		last_ob = series.values[index]
		inv_diff = inverse_difference(last_ob, inv_scale)
		# store
		inverted.append(inv_diff)
	return inverted
 
# evaluate the RMSE for each forecast time step
def evaluate_forecasts(test, forecasts, n_lag, n_seq):
	for i in range(n_seq):
		actual = [row[i] for row in test]
		predicted = [forecast[i] for forecast in forecasts]
		rmse = sqrt(mean_squared_error(actual, predicted))
		print('t+%d RMSE: %f' % ((i+1), rmse))
 
# plot the forecasts in the context of the original dataset
def plot_forecasts(series, forecasts, n_test):
	# plot the entire dataset in blue
	plt.plot(series.values)
	# plot the forecasts in red
	for i in range(len(forecasts)):
		off_s = len(series) - n_test + i - 1
		off_e = off_s + len(forecasts[i]) + 1
		xaxis = [x for x in range(off_s, off_e)]
		yaxis = [series.values[off_s]] + forecasts[i]
		plt.plot(xaxis, yaxis, color='red')
	# show the plot
	plt.show()
 
def create_sincos():
	t = np.linspace(0.0, np.pi*2.0, 100)
	n = 1000
	X = []
	X1 = []
	X2 = []
	for i in xrange(n):
	    if i<1:
	        x1 = np.cos(t) + np.random.normal(-0.04, 0.04, np.shape(t) ) #random.normal(mu, sig, )
	        x2 = np.sin(t) + np.random.normal(-0.04, 0.04, np.shape(t) )
	    elif i<100:
	        x1 = np.cos(t) + np.random.normal(-0.015, 0.015, np.shape(t) )
	        x2 = np.sin(t) + np.random.normal(-0.015, 0.015, np.shape(t) )        
	    elif i<200:
	        x1 = 0.9*np.cos(t) + np.random.normal(-0.025, 0.025, np.shape(t) )
	        x2 = 0.9*np.sin(t) + np.random.normal(-0.025, 0.025, np.shape(t) )        
	    elif i<300:
	        x1 = np.cos(1.1*t) + np.random.normal(-0.05, 0.05, np.shape(t) ) - 0.03
	        x2 = np.sin(1.1*t) + np.random.normal(-0.05, 0.05, np.shape(t) ) - 0.03       
	    elif i<400:
	        x1 = np.cos(0.8*t) + np.random.normal(-0.03, 0.03, np.shape(t) ) + 0.03
	        x2 = np.sin(0.8*t) + np.random.normal(-0.03, 0.03, np.shape(t) ) + 0.03        
	    elif i<500:
	        x1 = 1.1*np.cos(t) - 0.01
	        x2 = 1.1*np.sin(t) - 0.01     
	    elif i<600:
	        x1 = 1.13*np.cos(t) + np.random.normal(-0.02, 0.02, np.shape(t) ) - 0.02
	        x2 = 1.13*np.sin(t) + np.random.normal(-0.02, 0.02, np.shape(t) ) - 0.02       
	    elif i<700:
	        x1 = 1.15*np.cos(0.9*t) + np.random.normal(-0.01, 0.01, np.shape(t) )
	        x2 = 1.15*np.sin(0.9*t) + np.random.normal(-0.01, 0.01, np.shape(t) )        
	    elif i<800:
	        x1 = 1.12*np.cos(0.9*t) + np.random.normal(-0.035, 0.035, np.shape(t) )
	        x2 = 1.12*np.sin(0.9*t) + np.random.normal(-0.035, 0.035, np.shape(t) )        
	    elif i<900:
	        x1 = np.cos(t) 
	        x2 = np.sin(t)         
	    elif i<1000:
	        x1 = 1.1*np.cos(t) + 0.01
	        x2 = 1.1*np.sin(t) + 0.01        
	    # x1 = np.cos(t)
	    # x2 = np.sin(t)
	    X1.append(x1)
	    X2.append(x2)

	X1 = np.array(X1)
	X2 = np.array(X2)
	print 'x1 x2 shape'
	print X1.shape, X2.shape
	X.append(X1)
	X.append(X2)
	X = np.array(X)
	print X.shape

	# X1c = X1[0]
	# X2c = X2[0]
	# for i in range(1, len(X1)):
	#     X1c = np.concatenate((X1c, X1[i]))#, axis=0)
	#     X2c = np.concatenate((X2c, X2[i]))#, axis=0)
	# # print X1c.shape, X2c.shape

	# X = np.vstack((X1c, X2c))
	# print 'xshape'
	# print X.shape

	#Plot - Sanity Check
	# t = np.linspace(0.0, np.pi*2.0, 100*1000)
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
	ax01.plot(t, X[0,0,:])
	ax02.grid(True)
	ax02.plot(t, X[1,0,:])
	plt.show()

	#Pack into correct dataset shape
	# print 'packing'
	# print X.shape
	dataset = X
	dataset = dataset.astype('float32')

	# normalize the dataset
	# scaler = MinMaxScaler(feature_range=(0, 1))
	# dataset = scaler.fit_transform(dataset)
	# print 'dataset shape'
	# print dataset.shape
	return dataset

def prepare_sincos_data(data, n_in, n_out):
	print 'prepare sincos data'
	print data.shape
	# window over 100 datapoints for 1000samples
	# Shape:: (n, Batch=1000, Window=10, dim=1 or 2) -> 4D shape


	return scaler, train, test


def main():
	# fix random seed for reproducibility
	np.random.seed(7)

	# load dataset
	series = read_csv('./data/shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
	print 'shampoo shape'
	print series.shape
	print type(series)

	# dataset = create_sincos()
	# dataset = dataset[0] #1dimensional - just cosine
	# series = Series(dataset)

	# configure
	n_lag = 10
	n_seq = 10
	n_test = 15
	n_epochs = 5
	n_batch = 1
	n_neurons = 1
	# prepare data
	scaler, train, test = prepare_data(series, n_test, n_lag, n_seq) 
	# scaler, train, test = prepare_sincos_data(dataset, n_lag, n_seq)
	# fit model
	model = fit_lstm(train, n_lag, n_seq, n_batch, n_epochs, n_neurons)
	# make forecasts
	forecasts = make_forecasts(model, n_batch, train, test, n_lag, n_seq)
	# inverse transform forecasts and test
	forecasts = inverse_transform(series, forecasts, scaler, n_test+2)
	actual = [row[n_lag:] for row in test]
	actual = inverse_transform(series, actual, scaler, n_test+2)
	# evaluate forecasts
	evaluate_forecasts(actual, forecasts, n_lag, n_seq)
	# plot forecasts
	plot_forecasts(series, forecasts, n_test+2)

if __name__ == "__main__":
	main()
