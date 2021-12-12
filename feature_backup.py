import scipy.io
import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal
import pyrem as pr

def bandpower(x, fs, fmin, fmax):
    f, Pxx = scipy.signal.periodogram(x, fs=fs,scaling = 'spectrum')
    ind_min = np.argmax(f > fmin) - 1
    ind_max = np.argmax(f > fmax) - 1
    return np.trapz(Pxx[ind_min: ind_max], f[ind_min: ind_max])

def extract(Data,sample_freq,time,detec = 1):
	mean = np.mean(Data,axis = 1)
	maximum = np.amax(Data,axis = 1)
	minimum = np.amin(Data,axis = 1)
	stdv = np.std(Data,axis = 1)

	if detec == 1:
		freq_band = [0.1,4,8,12,30,70,180]
		ap = []
		for j in range(16):
			Dj = Data[j]
			for f in range(len(freq_band)-1):
				tmp = bandpower(Dj,sample_freq,freq_band[f],freq_band[f+1])
				ap.append(np.log(tmp))
	else:

		_,ap = scipy.signal.periodogram(Data,fs = sample_freq,nfft = 10)

		ap = list(ap.flatten())


	fft = scipy.fft.fft(Data,4).real.flatten()
	tmpf = [1,time]+list(mean)+list(maximum - minimum)+ap+list(fft)
	return np.array(tmpf)

def detecData(observer):

	foldername = '..\\detection\\'+observer
	files = os.listdir(foldername)
	y = []
	X = []
	N = len(files) # number of data segments
	for i in range(N): 
		fn = files[i]

		if 'test' in fn:
			continue

		mat = scipy.io.loadmat(foldername+'\\'+fn)
		k = list(mat.keys())
		emer = -1

		if len(k) == 7: # ictal
			y.append(-1)
			emer = mat[k[6]][0]
		else: # interictal
			y.append(1)

		X.append(extract(mat[k[3]],mat[k[4]],emer))
		print('processing file %d'%(i+1))


	X = np.array(X)
	y = np.array(y)

	xn = 'D_' + observer + 'x.csv'
	yn = 'D_' + observer + 'y.csv'

	np.savetxt(xn, X, delimiter=",")
	np.savetxt(yn, y, delimiter=",")

def predictData(observer):
	
	foldername = '..\\prediction\\'+observer
	files = os.listdir(foldername)

	y = []
	X = []
	N = len(files) # number of data segments

	for i in range(N):
		fn = files[i]

		if 'test' in fn:
			continue


		mat = scipy.io.loadmat(foldername+'\\'+fn)
		k = list(mat.keys())

		D = mat[k[3]][0][0][0]
		fs = mat[k[3]][0][0][2]
		s = mat[k[3]][0][0][4][0][0]

		w = len(D[0])//10
		for q in range(9):
			Ds = D.T[w*q:w*(q+1)].T
			X.append(extract(Ds,fs,s+q*0.1,0))


		Ds = D.T[9*w:].T
		X.append(extract(Ds,fs,s+9*0.1,0))

		if 'preictal' in k[3]: # preictal
			y += [-1] * 10
		else: # interictal
			y += [1] * 10
		

		print('processing file %d'%(i+1))

	X = np.array(X)
	y = np.array(y)

	xn = 'P_' + observer + 'x.csv'
	yn = 'P_' + observer + 'y.csv'

	np.savetxt(xn, X, delimiter=",")
	np.savetxt(yn, y, delimiter=",")


def checkP():
	foldn = '..\\prediction\\Dog_1'

	files = os.listdir(foldn)

	# prediction
	# ['__header__', '__version__', '__globals__', 'interictal_segment_1']
	mat = scipy.io.loadmat(foldn+'\\'+files[0])
	k = list(mat.keys())
	print(k)
	print(mat[k[3]][0][0][0].shape)

def checkD():
	foldn = '..\\detection\\Dog_1'
	files = os.listdir(foldn)

	# detection
	# ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels', 'latency'] -- ictal
	# OR
	# ['__header__', '__version__', '__globals__', 'data', 'freq', 'channels'] -- interictal
	mat = scipy.io.loadmat(foldn+'\\'+files[2000])
	k = list(mat.keys()) # 
	print(k)
	print(mat[k[3]].shape)



# transform all the dataset
def transform():
		
	for i in range(1,5):
		observer = 'Dog_%d'%i
		detecData(observer)
	for i in range(1,9):
		observer = 'Patient_%d'%i
		detecData(observer)

	for i in range(1,6):
		observer = 'Dog_%d'%i
		predictData(observer)
	for i in range(1,3):
		observer = 'Patient_%d'%i
		predictData(observer)


transform()