def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	# df = data
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


a = [1,2,3,4,5,6,7,8,9,10]
print series_to_supervised(a)

for j in xrange(len(x[0])-timesteps+1): # per window
    ## np.random.seed(3334 + i*len(x[0]) + j)                        
    ## noise = np.random.normal(0, noise_mag, (batch_size, timesteps, nDim))

    p = float(j)/float(length-timesteps+1) *2.0*phase - phase
    tr_loss = vae_autoencoder.train_on_batch(
        np.concatenate((x[:,j:j+timesteps], p*np.ones((len(x), timesteps, 1))), axis=-1), #-1 is the last column
        x[:,j:j+timesteps])

    seq_tr_loss.append(tr_loss)
mean_tr_loss.append( np.mean(seq_tr_loss) )
vae_autoencoder.reset_states()

# import numpy as np
# b = [ [[1],[1],[1]], [[2],[2],[2]], [[3],[3],[3]] ]
# b = np.array(b)
# print b.shape
# c = b
# print c.shape
# d = b
# d = np.concatenate((c,d), axis=-1)
# print d.shape
# print d

