import numpy as np

'''
Input:
	weight
	Data
	label 
	C = trade off of margin for unseperatable
	r1,r2 = extra weight for imbalanced data
Output:
	Loss -- type: double
'''
def Loss(weight, Data, label, c, r1, r2):

	N = len(label)

	yhat = Data.dot(weight)
	Dis = 1 - label*yhat

	Dis[label < 0] *= r1
	Dis[label > 0] *= r2

	L = np.sum(Dis[Dis > 0]) * c / N
	L += 0.5*(weight.dot(weight))
	
	return L

'''
Gradient of Loss function
	Same Inputs

Oputput:
	modifier of weight at each data point
		shape: p x 1
'''
def LossGrad(weight, xi, yi, c,r1,r2):

	yhat = xi.dot(weight)
	dis = 1 - yi*yhat

	'''
	res = weight
	if dis <= 0:
		return res
	
	tmp = c*yi*xi
	if yi < 0:
		return res - r1*tmp

	return res - r2*tmp
	'''
	res = np.zeros(weight.shape)
	if dis <= 0:
		return res
	
	tmp = c*yi*xi
	if yi < 0:
		return res - r1*tmp

	return res - r2*tmp

'''
Input:
	Data
	labels
	C = trade off of margin for unseperatable
	r1,r2 = extra weight for imbalanced data
	eta = learning rate (typically 10^-3 or 10^-4)
Oputput:
	w = weight
'''
def Mysvm(Data,y,C,r1,r2,eta):

	N,p = Data.shape
	weight = np.random.uniform(0,1,p)

	err = 1
	lO = 0
	MaxIter = 200
	k = 0
	while (err > 10**(-4) and k < MaxIter):

		for i in range(N):
			weight = weight - eta*LossGrad(weight,Data[i],y[i],C,r1,r2)
		l = Loss(weight,Data,y,C,r1,r2)
		err = abs(l-lO)
		lO = l
		k += 1

	return weight

