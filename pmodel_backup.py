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


'''
Input:
	l = actual label
	lh = estimate label
Output:
	TP = num(lh = -1 & l = -1) 
	TN = num(lh = 1 & l = 1)
	FP = num(lh = 1 & l = -1)
	FN = num(lh = -1 & l = 1)
Use -1 for positive is because 
	the preictal data is labeled as -1
'''

def ana(l,lh):
	res = [0,0,0,0]

	for i in range(len(l)):
		if lh[i] == -1:
			if l[i] == -1:
				res[0] += 1
			else: 
				res[3] += 1
		elif l[i] == 1:
			res[1] += 1
		else: 
			res[2] += 1 

	return res



def val(lres):
	TP,TN,FP,FN = lres

	if TP + FN == 0:
		sens = 0
	else:
		sens = TP/(TP+FN)
	F1 = TP/(TP + 0.5*(FP+FN))
	acc = (TP+TN)/(TP+TN+FP+FN)

	graph = np.array([[TP, FN],[FP,TN]])/(TP+TN+FP+FN)

	return sens, F1, acc, graph



observer = 'Dog_1'
X,y = loadData(observer,0)

NumIc = len(y[y>0])
NumInter = len(y) - NumIc

state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(y)

N,p = X.shape

# seperate data for cross-validation
Vx = []
Vy = []
n = N//5

for i in range(4):
	Xi = X[i*n:(i+1)*n][:]
	yi = y[i*n:(i+1)*n]
	Vx.append(Xi)
	Vy.append(yi)

Vx.append(X[4*n:])
Vy.append(y[4*n:])


# set possible C values
C = []
# r1 fixed at 1, change r2
R2 = []
eta = 10**(-4) # fixed
for i in range(5):
	C.append(2**(i))
	R2.append(0.1*(i+1))

SenG = np.zeros((5,5))


for p in range(5):
	xv = Vx[p]
	yv = Vy[p]
	ind = [0,1,2,3,4]
	ind.remove(p)

	xt = np.concatenate((Vx[ind[0]],Vx[ind[1]],Vx[ind[2]],Vx[ind[3]]),axis = 0)
	yt = np.concatenate((Vy[ind[0]],Vy[ind[1]],Vy[ind[2]],Vy[ind[3]]))

	for i in range(5):
		c = C[i]
		for j in range(5):
			r2 = R2[j]

			w = Mysvm(xt,yt,c,1,r2,eta)
			yh = np.sign(xv.dot(w))

			tmp = ana(yv,yh)
			SenG[i][j] += val(tmp)[0]

			print('p = %d, c = %2e, r2 = %2e'%(p,c,r2))


SenG /= 5
optc = 0
optr = 0

for i in range(5):
	for j in range(5):
		if SenG[i][j] > SenG[optc][optr]:
			optc = i
			optr = j

optc = C[optc]
optr = R2[optr]

plt.figure()
sns.heatmap(SenG, xticklabels = R2, yticklabels = C, annot = True, cmap="YlGnBu")
plt.savefig('P_'+observer+'_CR.png')
print(optc,optr)

statre = np.zeros((4,))
statBas = np.zeros((4,))


for p in range(5):
	xv = Vx[p]
	yv = Vy[p]
	ind = [0,1,2,3,4]
	ind.remove(p)

	xt = np.concatenate((Vx[ind[0]],Vx[ind[1]],Vx[ind[2]],Vx[ind[3]]),axis = 0)
	yt = np.concatenate((Vy[ind[0]],Vy[ind[1]],Vy[ind[2]],Vy[ind[3]]))

	w = Mysvm(xt,yt,optc,1,optr,eta)
	yh = np.sign(xv.dot(w))

	statre += np.array(ana(yv,yh))


	clf = svm.SVC()
	clf.fit(xt, yt)
	yh = clf.predict(xv)
	statBas += np.array(ana(yv,yh))



sens, F1, acc, graph = val(statre)

plt.figure()
labelx = ['preictal', 'interictal']
labely = ['preictal', 'interictal']
gr =  sns.heatmap(graph,xticklabels = labelx, yticklabels = labely, annot = True, cmap="YlGnBu")
plt.savefig('P_TPstretch'+observer+'.png')
textres = '%s\n\tCS-SVM:  Accuracy: %.4f, Sensitivity: %.4f, F1: %.4f\n\t\t\twith c = %.2f r2 = %.2f\n'\
	%(observer,acc,sens,F1,optc,optr)


sens, F1, acc, graph = val(statBas)

plt.figure()
labelx = ['preictal', 'interictal']
labely = ['preictal', 'interictal']
gr =  sns.heatmap(graph,xticklabels = labelx, yticklabels = labely, annot = True, cmap="YlGnBu")
plt.savefig('P_TPBase'+observer+'.png')
textres += '\t   SVM:  Accuracy: %.4f, Sensitivity: %.4f, F1: %.4f\n'%(acc,sens,F1)

f = open('P_'+observer+'_res.txt','w')
f.write(textres)
f.close()
