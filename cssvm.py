import numpy as np

'''
Input:
	weight, Data
	label = y
	C = trade off of margin for unseperatable
	R = extra weight for imbalanced data
Output:
	Loss -- type: double
'''
def Loss(weight, Data, label, C, R):

	N = len(label)

	yhat = Data.dot(weight)
	Dis = 1 - label*yhat

	Dis[label < 0] *= R

	L = np.sum(Dis[Dis > 0]) * C/ N
	L += 0.5*(weight.dot(weight))
	
	return L

'''
Gradient of Loss function
	Same Inputs

Oputput:
	modifier of weight
		shape: p x 1
'''
def LossGrad(weight, Data, label, C, R):

	yhat = Data.dot(weight)
	Dis = 1 - label*yhat

	N = len(Dis)

	res = np.zeros(weight.shape)

	for i in range(N):
		res += weight
		if Dis[i] <= 0:
			continue
		
		modi = C*label[i]*Data[i]
		if label[i] < 0:
			res -= R*modi
		else:
			res -= modi

	return res/N

'''
Input:
	X = Data
	y = labels
	C = trade off of margin for unseperatable
	R = extra weight for imbalanced data
	eta = learning rate (typically 10^-3 or 10^-4)
Oputput:
	w = weight
'''
def svm(X,y,C,R,eta):

	N,p = X.shape
	w = np.random.uniform(0,1,p)

	err = 1
	lO = 0
	MaxIter = 10**3 
	k = 0
	while (err > 10**(-4) and k < MaxIter):
		l = Loss(w,X,y,C,R)
		err = abs(l-lO)
		w -= eta*LossGrad(w,X,y,C,R)
		lO = l
		k += 1

	return w

