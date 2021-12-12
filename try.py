import numpy as np
from numpy import genfromtxt
from sklearn import svm
from cssvm import Mysvm
import seaborn as sns
import matplotlib.pyplot as plt


def loadData(ob,detec = 1):
	xn = ''
	yn = ''
	if detec == 1:
		xn = 'D_'+ob+'x.csv'
		yn = 'D_'+ob+'y.csv'
	else:
		xn = 'P_'+ob+'x.csv'
		yn = 'P_'+ob+'y.csv'

	return genfromtxt(xn, delimiter=','),genfromtxt(yn, delimiter=',')

observer = 'Patient_3'
X,y = loadData(observer)

print(X.shape)