import numpy as np
import math

def split():
	string = "0.67,-0.45,-1.85,-1.06,0.67,0.08,-0.85,-0.34,0.68,-0.24,0.88,-0.89,-0.26,-1.27,-0.13,-0.53,-0.43,0.10,0.06,0.23,-0.26,-0.11,-0.20,-0.36,-0.28,-0.82,-0.71,-0.29,-0.46,1.32,0.86,-1.66,0.14,-0.44,2.95,1.11,2.41,2.75,-1.28,-0.90,-0.73,-0.29,-1.01,-0.25,0.02,-0.33,-0.04,-0.23,0.91,1.23,1.20,1.04,0.29,0.37,0.28,0.20,0.24,-0.23,-0.55,-0.80,-1.72,-1.29,-1.24,-1.54,0.07,1.88,-1.23,-0.38,0.36,-1.67,-0.85,-0.97,0.67,-0.43,-1.18,-0.24,-0.27,-0.25,0.34,0.04,-0.09,-0.07,-1.18,-0.69,-1.16,-0.29,-0.23,-0.02,-0.53,-1.08,-0.13,-0.66,-0.41,-0.56,1.26,-0.39"
	temp = string.split(",")

	print(temp[1:])


	l = []
	l.append(temp)
	l.append(temp)



	a = 1
	b = 2
	c = 3

	m = np.zeros((2,96))

	for i in range(2):
		for j in range(96):
			m[i][j] = temp[j]

def lambdaSetup():
	l = np.ones((4,4))
	lamb = 2;
	l = l + l 
	l = np.diag(np.diag(l))
	print(l)



def testMatrixSegments():
	a = np.ones((4,4))
	b = np.ones((2,4))
	for i in range(2,4):
		b[i-2] = a[i]
		print(b[i-2])

def testMatrixSplice():
	a = np.ones((4,4))
	b = np.ones((4,1))
	a = np.delete(a,b,axis=0)
	print("\n")
	print(a)

def testMatrixDot():
	a = np.ones((4,4))
	b = np.ones((4,1))
	b = b + 3
	print(a)
	print(b)
	print(np.dot(a,b))
	c = np.multiply(a, b)
	print(c)
	c = np.dot(a,b)



testMatrixDot()

