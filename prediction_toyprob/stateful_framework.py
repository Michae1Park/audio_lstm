from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
from matplotlib import pyplot
from numpy import array
import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM, Dropout, RepeatVector, TimeDistributed, Input
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

import matplotlib.pyplot as plt
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation

#(1) generate data
#(2) window data
#(3) how to reshape, normalize, (optional) plot and check
#(4) train stateful -> make sure to reset properly, and call (1),(2),(3) use it like a generator
#(5) For prediction, first create an lstm with batchsize=1
#(6) create non-training data and predict
#(7) Plot and check

#Use these config only in main
BATCH_SIZE = 256
TIMESTEP_IN = 10
TIMESTEP_OUT = 10
INPUT_DIM = 1
NB_EPOCH = 100
N_NEURONS = 100

def main():
	# lstm_model = define_network(BATCH_SIZE, TIMESTEP_IN, INPUT_DIM, N_NEURONS)
	# lstm_model = fit_lstm(lstm_model)
	#predict
	new_model = Sequential()
	new_model.add(LSTM(N_NEURONS, batch_input_shape=(1, TIMESTEP_IN, INPUT_DIM), stateful=True))
	new_model.add(Dense(10))
	new_model.add(Activation('sigmoid')) #range [0,1], tanh=[-1,1]
	# old_weights = lstm_model.get_weights()
	# new_model.set_weights(old_weights)
	new_model.load_weights('./models/stateful.h5')
	new_model.compile(loss='mean_squared_error', optimizer='adam')
	
	dataset = []
	t = np.linspace(0.0, np.pi*2.0, 100)
	x1 = np.cos(t) 
	x2 = np.sin(t)
	dataset.append(x1)
	dataset.append(x2)
	dataset = np.array(dataset)
	print dataset.shape
	dataset = dataset[0] #[1,100]
	X, y = [], []
	for i in range(dataset.shape[0] - TIMESTEP_IN - TIMESTEP_OUT):
		X.append(dataset[i:i+TIMESTEP_IN])
		y.append(dataset[i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT])
	X = np.array(X)
	y = np.array(y)
	X = X.reshape(X.shape[0],1,10,1)
	print X.shape, y.shape
	X, vmax, vmin = normalize(X) #valid max min
	rst = []
	for i in range(X.shape[0]):
		tmp = new_model.predict(X[i], batch_size=1)
		rst.append(tmp)
	rst = np.array(rst)
	print rst.shape
	rst = scale_back(rst, vmin, vmax)
	X = scale_back(X, vmin, vmax)
	for i in range(rst.shape[0]):
		xaxis = [x for x in range(i, i+TIMESTEP_OUT)]
		pyplot.plot(xaxis, X[i,0,:,0], color='blue')
		xaxis2 = [x for x in range(i+TIMESTEP_IN, i+TIMESTEP_IN+TIMESTEP_OUT)]
		pyplot.plot( xaxis2 ,rst[i,0,:], color='red')
	#plot -> predicted and dataset and compare result
	# pyplot.plot(dataset)
	# for i in range(20, rst.shape[0]):
	# 	xaxis = [x for x in range(i, i+TIMESTEP_OUT)]
	# 	pyplot.plot( xaxis ,rst[i,0,:], color='red')
	pyplot.show()

def define_network(batch_size, timesteps, input_dim, n_neurons):
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(batch_size, timesteps, input_dim), stateful=True))
	model.add(Dense(10))
	model.add(Activation('sigmoid')) #range [0,1], tanh=[-1,1]
	model.compile(loss='mean_squared_error', optimizer='adam')
	return model

def fit_lstm(model):
	for i in range(NB_EPOCH):
		X, y = data_generator()
		#For 1D
		X = X[:,:,:,0]
		y = y[:,:,:,0]
		X = X.reshape(X.shape[0],X.shape[1],X.shape[2],1)
		print X.shape, y.shape
		X, dmax, dmin = normalize(X) #dummy max, dummy min
		y, dmax, dmin = normalize(y)
		print 'verbose training iteration' + str(i)
		for j in range(X.shape[0]):
			model.train_on_batch(X[j], y[j])
		model.reset_states() #reset for every epoch
	model.save_weights('./models/stateful.h5')
	return model

def scale_back(seq, min_y, max_y):
    # scale back 
    seq = seq * (max_y - min_y) + min_y
    return seq

def normalize(data):
    a_max = np.max(data)
    a_min =  np.min(data)
    data = (data - a_min) / (a_max - a_min)
    return data, a_max, a_min

def data_generator():
	dataset = generate_sincos() 
	X, y = format_data(dataset) # X.shape=(Batch=256,10,2), y.shape=(256,10,2)
	return X,y

def format_data(dataset): #dataset.shape=(batchsize=256, datapoints=100, dim=2)
	X, y = [], []
	for i in range(dataset.shape[1] - TIMESTEP_IN - TIMESTEP_OUT):
		X.append(dataset[:, i:i+TIMESTEP_IN, :])
		y.append(dataset[:, i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :])
	X = np.array(X)
	y = np.array(y)
	print 'format data?'
	print X.shape, y.shape
	return X, y

def generate_sincos():
	t = np.linspace(0.0, np.pi*2.0, 100)
	batch_size = 256
	n = 16
	X, X1, X2 = [], [], []
	for i in xrange(batch_size):
	    if i<n:
	        x1 = np.cos(t) + np.random.normal(-0.04, 0.04, np.shape(t) ) #random.normal(mu, sig, )
	        x2 = np.sin(t) + np.random.normal(-0.04, 0.04, np.shape(t) )
	    elif i<2*n:
	        x1 = np.cos(t) + np.random.normal(-0.015, 0.015, np.shape(t) )
	        x2 = np.sin(t) + np.random.normal(-0.015, 0.015, np.shape(t) )        
	    elif i<3*n:
	        x1 = 0.9*np.cos(t) + np.random.normal(-0.025, 0.025, np.shape(t) )
	        x2 = 0.9*np.sin(t) + np.random.normal(-0.025, 0.025, np.shape(t) )        
	    elif i<4*n:
	        x1 = np.cos(1.1*t) + np.random.normal(-0.05, 0.05, np.shape(t) ) - 0.03
	        x2 = np.sin(1.1*t) + np.random.normal(-0.05, 0.05, np.shape(t) ) - 0.03       
	    elif i<5*n:
	        x1 = np.cos(0.8*t) + np.random.normal(-0.03, 0.03, np.shape(t) ) + 0.03
	        x2 = np.sin(0.8*t) + np.random.normal(-0.03, 0.03, np.shape(t) ) + 0.03        
	    elif i<6*n:
	        x1 = 1.1*np.cos(t) - 0.01
	        x2 = 1.1*np.sin(t) - 0.01     
	    elif i<7*n:
	        x1 = 1.13*np.cos(t) + np.random.normal(-0.02, 0.02, np.shape(t) ) - 0.02
	        x2 = 1.13*np.sin(t) + np.random.normal(-0.02, 0.02, np.shape(t) ) - 0.02       
	    elif i<8*n:
	        x1 = 1.15*np.cos(0.9*t) + np.random.normal(-0.01, 0.01, np.shape(t) )
	        x2 = 1.15*np.sin(0.9*t) + np.random.normal(-0.01, 0.01, np.shape(t) )        
	    elif i<9*n:
	        x1 = 1.12*np.cos(0.9*t) + np.random.normal(-0.035, 0.035, np.shape(t) )
	        x2 = 1.12*np.sin(0.9*t) + np.random.normal(-0.035, 0.035, np.shape(t) )        
	    elif i<10*n:
	        x1 = np.cos(t) 
	        x2 = np.sin(t)         
	    elif i<11*n:
	        x1 = 1.1*np.cos(t) + 0.01
	        x2 = 1.1*np.sin(t) + 0.01    
	    elif i<12*n:
	        x1 = np.cos(t) + 0.01
	        x2 = np.sin(t) + 0.01
	    elif i<13*n:
	    	x1 = np.cos(t) + np.random.normal(-0.02, 0.02, np.shape(t) )
	    	x2 = np.sin(t) + np.random.normal(-0.02, 0.02, np.shape(t) )
	    elif i<14*n:
			x1 = 1.1*np.cos(0.9*t) + np.random.normal(-0.012, 0.012, np.shape(t) ) + 0.005
			x2 = 1.1*np.sin(0.9*t) + np.random.normal(-0.012, 0.012, np.shape(t) ) + 0.005		
	    elif i<15*n:
			x1 = np.cos(t) + np.random.normal(-0.023, 0.023, np.shape(t) ) - 0.005
			x2 = np.sin(t) + np.random.normal(-0.023, 0.023, np.shape(t) ) - 0.005
	    elif i<16*n:
			x1 = 0.95*np.cos(t) + np.random.normal(-0.018, 0.018, np.shape(t) )
			x2 = 0.95*np.sin(t) + np.random.normal(-0.018, 0.018, np.shape(t) )        
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
	X = np.rollaxis(X, 0, 3)
	print X.shape

	#Plot - Sanity Check
	# t = np.linspace(0.0, np.pi*2.0, 100)
	# font = {'size'   : 9}
	# matplotlib.rc('font', **font)
	# f0 = figure(num = 0, figsize = (12, 8))#, dpi = 100)
	# f0.suptitle("feature plotting", fontsize=12)
	# ax01 = subplot2grid((2, 1), (0, 0))
	# ax02 = subplot2grid((2, 1), (1, 0))
	# ax01.grid(True)
	# ax01.plot(t, X[0,:,0])
	# ax02.grid(True)
	# ax02.plot(t, X[0,:,1])
	# plt.show()

	X = X.astype('float32')
	return X

if __name__ == "__main__":
	main()

