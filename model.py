import numpy as np
from numpy import genfromtxt
from sklearn import svm
from sklearn import metrics
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


observer = 'Dog_4'
X,yprim = loadData(observer)


y = np.zeros(yprim.shape)

y[yprim<0] += 1


NumIc = len(y[y==1])
NumInter = len(y) - NumIc

state = np.random.get_state()
np.random.shuffle(X)
np.random.set_state(state)
np.random.shuffle(y)

N,p = X.shape
print(N)
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
	C.append((i+5)/10)
for i in range(5):
	C.append(i+1)
for i in range(10):
	R2.append((i+1)/10)

AUC = np.zeros((10,10))


for ci in range(10):
	c = C[ci]


	for ri in range(10):
		r = R2[ri]

		clf = svm.SVC(C = c, class_weight = {0:r, 1:1})
		for testi in range(5):

			Vxtest = Vx[testi]
			Vytest = Vy[testi]

			I = [0,1,2,3,4]
			
			I.remove(testi)

			Vyprob = np.zeros(Vytest.shape)

			for i in range(4):

				Vxvalid = Vx[I[i]]
				Vyvalid = Vy[I[i]]

				trainI = []
				for tmp in I:
					trainI.append(tmp)
				trainI.remove(I[i])


				Vxtrain = np.concatenate((Vx[trainI[0]],Vx[trainI[1]],Vx[trainI[2]]),axis = 0)
				Vytrain = np.concatenate((Vy[trainI[0]],Vy[trainI[1]],Vy[trainI[2]]))

				clf.fit(Vxtrain,Vytrain)
				
				Vyprob += clf.predict(Vxtest)

			Vyprob /= 4
			auc = metrics.roc_auc_score(Vytest, Vyprob)
			AUC[ci][ri] += auc

			print('testi = %d, c = %2e, r2 = %2e, auc ='%(testi,c,r),auc)

AUC /= 5

plt.figure()
sns.heatmap(AUC, xticklabels = R2, yticklabels = C, annot = True, cmap="YlGnBu")
plt.savefig('Dres\\D_'+observer+'_CR.png')

ci,ri = np.unravel_index(np.argmax(AUC, axis=None), AUC.shape)

optc = C[ci]
optr = R2[ri]

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

'''
TP = G[0][0]
FN = G[0][1]
FP = G[1][0]
TN = G[1][1]
'''
G = np.zeros((2,2))
Gb = np.zeros((2,2))

for i in range(5):

	Vxvalid = Vx[i]
	Vyvalid = Vy[i]

	I = [0,1,2,3,4]
	I.remove(i)

	Vxt = np.concatenate((Vx[I[0]],Vx[I[1]],Vx[I[2]],Vx[I[3]]),axis = 0)
	Vyt = np.concatenate((Vy[I[0]],Vy[I[1]],Vy[I[2]],Vy[I[3]]))

	clf = svm.SVC(C = optc, class_weight = {0:optr, 1:1})
	clf.fit(Vxt,Vyt)
	yhat = clf.predict(Vxvalid)


	clfbase = svm.SVC()
	clfbase.fit(Vxt,Vyt)
	ybase = clfbase.predict(Vxvalid)

	for k in range(len(yhat)):
		if Vyvalid[k] == 1:
			if yhat[k] == Vyvalid[k]:
				G[0][0] += 1
			else: 
				G[1][0] += 1

			if ybase[k] == Vyvalid[k]:
				Gb[0][0] += 1
			else: 
				Gb[1][0] += 1

		else:
			if yhat[k] == Vyvalid[k]:
				G[1][1] += 1
			else:
				G[0][1] += 1

			if ybase[k] == Vytrain[k]:
				Gb[1][1] += 1
			else: 
				Gb[0][1] += 1


plt.figure()
labelx = ['ictal', 'interictal']
labely = ['ictal', 'interictal']
gr =  sns.heatmap(G/N,xticklabels = labelx, yticklabels = labely, annot = True, cmap="YlGnBu")
plt.xlabel('actual')
plt.ylabel('predict')
plt.title(observer+'_CSSVM')
plt.savefig('Dres\\D_TPstretch'+observer+'.png')

plt.figure()
gr =  sns.heatmap(Gb/N,xticklabels = labelx, yticklabels = labely, annot = True, cmap="YlGnBu")
plt.xlabel('actual')
plt.ylabel('predict')
plt.title(observer+'_SVM')
plt.savefig('Dres\\D_TPBase'+observer+'.png')

[[TP,FN],[FP,TN]] = G
acc = (TP+TN)/N
sensivity = 0
if TP != 0:
	sensivity = TP/(TP+FN)
F1 = 2*TP/(2*TP + FP + FN)

textres = '%s -- {ictal:interictal} = {%.2f:%.2f}\n'%(observer,NumIc/N,NumInter/N)


textres += '\tCS-SVM:  Accuracy: %.4f, Sensitivity: %.4f, F1: %.4f\n\t\t\twith c = %.2f r2 = %.2f\n'\
	%(acc,sensivity,F1,optc,optr)

[[TP,FN],[FP,TN]] = Gb
acc = (TP+TN)/N
sensivity = 0
if TP != 0:
	sensivity = TP/(TP+FN)
F1 = 2*TP/(2*TP + FP + FN)

textres += '\t   SVM:  Accuracy: %.4f, Sensitivity: %.4f, F1: %.4f\n'%(acc,sensivity,F1)

f = open('Dres\\D_'+observer+'_res.txt','w')
f.write(textres)
f.close()
