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
from keras.optimizers import Adam
from keras import optimizers
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras import backend as K
import math
from matplotlib.pylab import *
from mpl_toolkits.axes_grid1 import host_subplot
import matplotlib.animation as animation
from sklearn.model_selection import train_test_split
import gc

#Use these config only in main
BATCH_SIZE = 1
PRED_BATCH_SIZE = 1
TIMESTEP_IN = 2
TIMESTEP_OUT = 10
INPUT_DIM = 1
NB_EPOCH = 500
N_NEURONS = TIMESTEP_OUT
TEST_SHIFT = 0
LOAD_WEIGHT = True
WEIGHT_FILE = './models/stateful-OneToMany-tanh2.h5'
PLOT = True
NUM_BATCH = 100 #Total #samples = Num_batch x Batch_size

def data_generator():
	dataset = generate_sincos() 
	X, y = format_data(dataset) # X.shape=(Batch=256,10,2), y.shape=(256,10,2)
	return X,y #(num_window, batch, window_size, dim)

def format_data(dataset): #dataset.shape=(batchsize=256, datapoints=100, dim=2)
	X, y = [], []
	for i in range(dataset.shape[1] - TIMESTEP_IN - TIMESTEP_OUT + 1):
		x_f = dataset[:, i:i+TIMESTEP_IN, :]
		y_f = dataset[:, i+TIMESTEP_IN:i+TIMESTEP_IN+TIMESTEP_OUT, :]
		X.append(x_f)
		y.append(y_f)
	X = np.array(X)
	y = np.array(y)
	print 'windowed data'
	print X.shape, y.shape
	return X, y

def generate_sincos():
	t = np.linspace(0.0, np.pi*2.0, 100)
	batch_size = BATCH_SIZE #32,64,128,256,512,1024,2048
	n = batch_size/16 
	X, X1, X2 = [], [], []
	for j in range(NUM_BATCH): 
		for i in xrange(batch_size):
			# x1 = 0.8*np.cos(t) 
			# x2 = 0.8*np.sin(t) 
			if i<n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.04, 0.04, np.shape(t) ) #random.normal(mu, sig, )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.04, 0.04, np.shape(t) )
			elif i<2*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.015, 0.015, np.shape(t) )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.015, 0.015, np.shape(t) )        
			elif i<3*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.025, 0.025, np.shape(t) )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.025, 0.025, np.shape(t) )        
			elif i<4*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.05, 0.05, np.shape(t) ) #- 0.03
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.05, 0.05, np.shape(t) ) #- 0.03       
			elif i<5*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.03, 0.03, np.shape(t) ) #+ 0.03
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.03, 0.03, np.shape(t) ) #+ 0.03        
			elif i<6*n:
				x1 = 0.8*np.cos(t) #- 0.01
				x2 = 0.8*np.sin(t) #- 0.01     
			elif i<7*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.02, 0.02, np.shape(t) ) #- 0.02
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.02, 0.02, np.shape(t) ) #- 0.02       
			elif i<8*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.01, 0.01, np.shape(t) )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.01, 0.01, np.shape(t) )        
			elif i<9*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.035, 0.035, np.shape(t) )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.035, 0.035, np.shape(t) )        
			elif i<10*n:
				x1 = 0.8*np.cos(t) 
				x2 = 0.8*np.sin(t)         
			elif i<11*n:
				x1 = 0.8*np.cos(t) #+ 0.01
				x2 = 0.8*np.sin(t) #+ 0.01    
			elif i<12*n:
				x1 = 0.8*np.cos(t) #+ 0.01
				x2 = 0.8*np.sin(t) #+ 0.01
			elif i<13*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.02, 0.02, np.shape(t) )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.02, 0.02, np.shape(t) )
			elif i<14*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.012, 0.012, np.shape(t) ) #+ 0.05
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.012, 0.012, np.shape(t) ) #+ 0.05      
			elif i<15*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.023, 0.023, np.shape(t) ) #- 0.05
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.023, 0.023, np.shape(t) ) #- 0.05
			elif i<16*n:
				x1 = 0.8*np.cos(t) #+ np.random.normal(-0.018, 0.018, np.shape(t) )
				x2 = 0.8*np.sin(t) #+ np.random.normal(-0.018, 0.018, np.shape(t) )        
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

	# for i in range(batch_size):
	#   pyplot.plot(X[i,:,0])
	# pyplot.show()

	X = X.astype('float32')
	if INPUT_DIM==1:
		X = X[:,:,0] #0=cos, 1=sin
		X = X.reshape(X.shape[0], X.shape[1], 1)
		print X.shape
	print 'end of generate_sincos func'

	return X

def define_network(batch_size, timesteps, input_dim, n_neurons, load_weight=False):
	model = Sequential()
	model.add(LSTM(n_neurons, batch_input_shape=(batch_size, timesteps, input_dim),
					stateful=False, activation='tanh'))
	if load_weight:
		model.load_weights(WEIGHT_FILE)
	#optimizer = RMSprop(lr=0.001)#, rho=0.9, epsilon=1e-08, decay=0.0001, clipvalue=10)
	# optimizer = Adam(lr=0.005)#lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
	optimizer = optimizers.Adam(lr=0.001)
	loss = 'mean_squared_error'
	model.compile(loss=loss, optimizer=optimizer)
	print model.summary()
	return model

def fit_lstm(model, x_train, x_test, y_train, y_test):
	wait         = 0
	plateau_wait = 0
	min_loss = 1e+15
	patience = 5
	plot_tr_loss = []
	plot_te_loss = []
	for epoch in range(NB_EPOCH):
		#train
		mean_tr_loss = []
		for i in range(0,x_train.shape[0], BATCH_SIZE):
			#per window
			seq_tr_loss = []
			x = x_train[i:i+BATCH_SIZE]
			y = y_train[i:i+BATCH_SIZE]
			x = np.swapaxes(x, 0, 1)
			y = np.swapaxes(y, 0, 1)
			for j in range(x.shape[0]):
				tr_loss = model.train_on_batch(x[j], y[j])
				seq_tr_loss.append(tr_loss)
			mean_tr_loss.append( np.mean(seq_tr_loss) )
			model.reset_states()
		tr_loss = np.mean(mean_tr_loss)
		sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\n'.format(epoch, NB_EPOCH, tr_loss, 0))
		sys.stdout.flush()
		plot_tr_loss.append(tr_loss)

		#test  
		mean_te_loss = []
		for i in xrange(0, x_test.shape[0], BATCH_SIZE):
			seq_te_loss = []
			x = x_test[i:i+BATCH_SIZE]
			y = y_test[i:i+BATCH_SIZE]
			x = np.swapaxes(x, 0, 1)
			y = np.swapaxes(y, 0, 1)
			for j in xrange(x.shape[0]):
				te_loss = model.test_on_batch(x[j], y[j])
				seq_te_loss.append(te_loss)
			mean_te_loss.append( np.mean(seq_te_loss) )
			model.reset_states()
		val_loss = np.mean(mean_te_loss)
		sys.stdout.write('Epoch {} / {} : loss training = {} , loss validating = {}\n'.format(epoch, NB_EPOCH, tr_loss, val_loss))
		sys.stdout.flush()   
		plot_te_loss.append(val_loss)

		# Early Stopping
		if val_loss <= min_loss:
			min_loss = val_loss
			wait         = 0
			plateau_wait = 0
			print 'saving model'
			model.save_weights(WEIGHT_FILE) 
		else:
			if wait > patience:
				print "Over patience!"
				break
			else:
				wait += 1
				plateau_wait += 1

		#ReduceLROnPlateau
		if plateau_wait > 2:
			old_lr = float(K.get_value(model.optimizer.lr)) #K is a backend
			new_lr = old_lr * 0.2
			K.set_value(model.optimizer.lr, new_lr)
			plateau_wait = 0
			print 'Reduced learning rate {} to {}'.format(old_lr, new_lr)

		gc.collect()    

	# ---------------------------------------------------------------------------------
	# visualize outputs
	print "Training history"
	fig = pyplot.figure(figsize=(10,4))
	ax1 = fig.add_subplot(1, 2, 1)
	pyplot.plot(plot_tr_loss)
	ax1.set_title('loss')
	ax2 = fig.add_subplot(1, 2, 2)
	pyplot.plot(plot_te_loss)
	ax2.set_title('validation loss')
	pyplot.show()

	return model

def predict(new_model):
	dataset = []
	t = np.linspace(0.0, np.pi*2.0, 100)
	# Test1 -- slight variation, if work well then add offset
	# x1 = 0.8*np.cos(t+TEST_SHIFT) + np.random.normal(-0.033, 0.033, np.shape(t) ) #+ 0.05
	# x2 = 0.8*np.sin(t+TEST_SHIFT) + np.random.normal(-0.033, 0.033, np.shape(t) ) #+ 0.05
	# Test2
	x1 = 0.75*np.cos(t) + np.random.normal(-0.03, 0.03, np.shape(t) )
	x2 = 0.75*np.sin(t) + np.random.normal(-0.03, 0.03, np.shape(t) )
	x2 = np.concatenate(([0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75], x2))
	x1 = np.concatenate(([0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75,0.75], x1))
	# Test3 -- same as training
	# x1 = 0.8*np.cos(t)
	# x2 = 0.8*np.sin(t)
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
	X = X.reshape(X.shape[0], PRED_BATCH_SIZE, TIMESTEP_IN, INPUT_DIM)
	rst = []
	# x_tmp = X[0]
	for i in range(0, X.shape[0]):
		p_tmp = new_model.predict_on_batch(X[i]) #, batch_size=PRED_BATCH_SIZE)
		rst.append(p_tmp)
	rst = np.array(rst)
	print rst.shape

	#PLOT
	# pyplot.plot(X[:,0,:,0]) #Original Plot for OneToMany
	# pyplot.plot(X[:,:]) #Dense Plot
	for i in range(0, rst.shape[0]):
		xaxis = [x for x in range(i, i+TIMESTEP_IN)]
		pyplot.plot(xaxis, X[i,0,:,0], color='blue')
		# pyplot.plot(xaxis, X[i,0,:], color='blue') #2D Dense Plot
		xaxis2 = [x for x in range(i+TIMESTEP_IN, i+TIMESTEP_IN+TIMESTEP_OUT)]
		pyplot.plot(xaxis2 ,rst[i,0,:], color='red')
		# pyplot.plot(xaxis2 ,rst[i,0,:], color='red') #2D Dense Plot
	pyplot.show()

def main():
	'''
	dataset.shape:: (num_window, batch x N, window_size, dim)
	'''
	X, y = data_generator() #generates entire dataset to be used shape is listed above
	X = X.reshape(X.shape[0],X.shape[1],X.shape[2],INPUT_DIM)
	y = y.reshape(y.shape[0],y.shape[1],y.shape[2])#,INPUT_DIM)
	X = np.swapaxes(X, 0, 1)
	y = np.swapaxes(y, 0, 1)
	print 'in main'
	print X.shape, y.shape
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
	print X_train.shape, X_test.shape, y_train.shape, y_test.shape

	# np.random.seed(3334)
	#train phase
	lstm_model = define_network(BATCH_SIZE, TIMESTEP_IN, INPUT_DIM, N_NEURONS, False)
	# lstm_model = fit_lstm(lstm_model, X_train, X_test, y_train, y_test)
	#predict phase
	new_model = define_network(PRED_BATCH_SIZE, TIMESTEP_IN, INPUT_DIM, N_NEURONS, LOAD_WEIGHT)
	predict(new_model)

if __name__ == "__main__":
	main()

