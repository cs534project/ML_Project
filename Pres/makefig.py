from PIL import Image

def nameList(Func):
	l = []

	for i in range(1,6):
		l.append('P_TP%sDog_%d.png'%(Func,i))
	for i in range(1,3):
		l.append('P_TP%sPatient_%d.png'%(Func,i))

	return l

def nameLCR():
	l = []

	for i in range(1,6):
		l.append('P_Dog_%d_CR.png'%(i))
	for i in range(1,3):
		l.append('P_Patient_%d_CR.png'%(i))

	return l


def sumPart(f):
	if f != '':
		imL = [Image.open(fn) for fn in nameList(f)]
	else:
		imL = [Image.open(fn) for fn in nameLCR()]

	width,height = imL[0].size

	result = Image.new(imL[0].mode,(width,height*len(imL)))

	for i,im in enumerate(imL):
		result.paste(im,box=(0,i*height))

	if f != '':
		f += '_'
	result.save(f+'Summary.png')

def com():
	imN = ['Base_Summary.png','stretch_Summary.png','Summary.png']
	imL = [Image.open(fn) for fn in imN]

	width,height = imL[0].size
	result = Image.new(imL[0].mode,(width*3,height))

	for i,im in enumerate(imL):
		result.paste(im,box=(i*width,0))

	result.save('Compare.png')

sumPart('Base')
sumPart('stretch')
sumPart('')
com()