import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# def sample(preds, temperature=1.0, min_value=0, max_value=1):
#     # helper function to sample an index from a probability array
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)				#change to log and back
#     preds = exp_preds / np.sum(exp_preds)	#normalize
#     probas = np.random.multinomial(1, preds, 1) #get some probability of preds?
#     v = np.argmax(probas)/float(probas.shape[1]) #
#     return v * (max_value - min_value) + min_value #normalize the probability?

def main():
	#audio_filename1 = './datasets/data1.wav'
	#audio_filename2 = './datasets/data2.wav'
	#audio_filename3 = './datasets/data5.wav'
	#book = './datasets/AudioBook.wav'
	audio_filename4 = './datasets/data20.wav'

	sr = 44100
	#b1, _ = librosa.load(book, sr=sr, mono=True)
	#y1, _ = librosa.load(audio_filename1, sr=sr, mono=True)
	#y2, _ = librosa.load(audio_filename2, sr=sr, mono=True)
	#y3, _ = librosa.load(audio_filename3, sr=sr, mono=True)
	y4, _ = librosa.load(audio_filename4, sr=sr, mono=True)

	#print y1.shape, y2.shape, y3.shape

	#amplify
	#y1 = y1
	#y2 = y2*3
	#y3 = y3*5
	y4 = y4*6

	#y1 = y1[55536:147696]
	#y2 = y2[0:92160]
	#y3 = y3[0:92160]
	#b1 = b1[100000:300000]
	y4 = y4[100000:192160]

	# min_y1 = np.min(y1)
	# max_y1 = np.max(y1)
	# y1 = (y1 - min_y1) / (max_y1 - min_y1)

	# min_y2 = np.min(y2)
	# max_y2 = np.max(y2)
	# y2 = (y2 - min_y2) / (max_y2 - min_y2)

	# min_y3 = np.min(y3)
	# max_y3 = np.max(y3)
	# y3 = (y3 - min_y3) / (max_y3 - min_y3)

	librosa.output.write_wav('./datasets/YourMusicLibrary/data20crop4Crop.wav', y4, sr)
	#librosa.output.write_wav('./datasets/YourMusicLibrary/AudioBookCrop.wav', b1, sr)
	#librosa.output.write_wav('./datasets/YourMusicLibrary/data1crop4.wav', y1, sr)
	#librosa.output.write_wav('./datasets/YourMusicLibrary/data2crop4.wav', y2, sr)
	#librosa.output.write_wav('./datasets/YourMusicLibrary/data5crop4.wav', y3, sr)

	# l = y1.shape[0]
	# l = l/6
	# #crop audio file
	# y1 = y1[0:l]

	# l = y2.shape[0]
	# l = l/6
	# #crop audio file
	# y2 = y2[0:l]

	# l = y3.shape[0]
	# l = l/6
	# #crop audio file
	# y3 = y3[0:l]

	# l = y4.shape[0]
	# l = l/6
	# #crop audio file
	# y4 = y4[0:l]

	# l = y5.shape[0]
	# l = l/6
	# #crop audio file
	# y5 = y5[0:l]

	# l = y6.shape[0]
	# l = l
	# #crop audio file
	# y6 = y6[0:l]

	# l = y7.shape[0]
	# l = l/6
	# #crop audio file
	# y7 = y7[0:l]

	# l = y8.shape[0]
	# l = l/6
	# #crop audio file
	# y8 = y8[0:l]
	

	# y = []
	# y.append(y1)
	# y.append(y2)
	# y.append(y3)
	# y.append(y4)
	# y.append(y5)
	# y.append(y6)
	# y.append(y7)
	# y.append(y8)
	# y = np.hstack(y)

	# librosa.output.write_wav('dorecont.wav', y, sr)
	# y = y5
	# #normalize
	# min_y = np.min(y)
	# max_y = np.max(y)
	# y = (y - min_y) / (max_y - min_y)
	# l = len(y)
	# #plot time series of audio data to see continuity
	# x1 = np.arange(l)
	# y1 = np.array(y)

	# print x1
	# print y1

	# plt.plot(x1,y1)
	# plt.show()


	# print 'shape:'
	# print y.shape
	# print y.shape[0]
	# print y
	# print type(y)
	# print type(y[1])
	# x = []
	# for i in range(0, 2000):
	# 	x.append(i)
	# y = np.array(x, dtype=float)
	# min_y = np.min(y)
	# max_y = np.max(y)
	# # normalize
	# y = (y - min_y) / (max_y - min_y)
	# #print y

	# maxlen = 128
	# nb_output = 256
	# step = 5
	# next_sample = []
	# samples = []

	# for j in range(0, y.shape[0] - maxlen, step):
	#     seq = y[j: j + maxlen + 1]  
	#     print seq
	#     print '\n'
	#     seq_matrix = np.zeros((maxlen, nb_output), dtype=bool) 
	#     for i,s in enumerate(seq):
	#         sample_ = int(s * (nb_output - 1)) # 0-255
	#         if i < maxlen:
	#             seq_matrix[i, sample_] = True
	#         else:
	#             seq_vec = np.zeros(nb_output, dtype=bool)
	#             seq_vec[sample_] = True
	#             print seq_vec
	#             print '\n'
	#             next_sample.append(seq_vec)
	#     samples.append(seq_matrix)
	# samples = np.array(samples, dtype=bool)
	# next_sample = np.array(next_sample, dtype=bool)
	# print samples.shape, next_sample.shape



	# librosa.output.write_wav('test.wav', y, sr)

if __name__ == '__main__':
	main()