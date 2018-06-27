import numpy as np
import math
import csv

# w = (p+1) x 1

#constants

train_filename = "spam-train"
test_filename = "spam-test"

NUM_TRAINING_INSTANCES = 3065
NUM_TESTING_INSTANCES = 1536

NUM_FEATURES = 57 + 1

#ALPHA = 0.0001
EPSILON = 0.01
EPSILON_TRANS = 0.00000001

#a = 0.00001
#

# a = 0.001
#


#functions
def S(r):
	return 1 / (1 + math.exp((-1)*r))

def sigmoid(r):
	for i in range(len(r)):
		r[i] = 1 / (1 + math.exp((-1)*r[i]))
	return r

def getTrainingData():
	rawTrainingData = open(train_filename,"r")
	lines = rawTrainingData.read().split("\n")

	temp_list = []
	y_list = []

	for i in range(NUM_TRAINING_INSTANCES):
		string = lines[i].split(",")
		y_list.append(string[NUM_FEATURES-1:])
		temp_list.append(string[:-1])

	x = np.ones((NUM_TRAINING_INSTANCES,NUM_FEATURES))
	y = np.ones((NUM_TRAINING_INSTANCES,1))

	for i in range(NUM_TRAINING_INSTANCES):
		y[i] = y_list[i]
		for j in range(NUM_FEATURES-1):
			x[i][j] = temp_list[i][j]

	return x,y

def getTestingData():
	rawTestingData = open(test_filename,"r")
	lines = rawTestingData.read().split("\n")

	temp_list = []
	y_list = []

	for i in range(NUM_TESTING_INSTANCES):
		string = lines[i].split(",")
		y_list.append(string[NUM_FEATURES-1:])
		temp_list.append(string[:-1])

	x = np.ones((NUM_TESTING_INSTANCES,NUM_FEATURES))
	y = np.ones((NUM_TESTING_INSTANCES,1))

	for i in range(NUM_TESTING_INSTANCES):
		y[i] = y_list[i]
		for j in range(NUM_FEATURES-1):
			x[i][j] = temp_list[i][j]

	return x,y

def standardize(x_train, x_test):

	#standardize training data
	for i in range(NUM_FEATURES-1):

		mean = np.mean(x_train.transpose()[i])
		sdev = np.std(x_train.transpose()[i])

		for j in range(NUM_TRAINING_INSTANCES):
			x_train[j][i] = (x_train[j][i] - mean)/sdev


	#standardize testing data
	for i in range(NUM_FEATURES-1):

		mean = np.mean(x_test.transpose()[i])
		sdev = np.std(x_test.transpose()[i])

		for j in range(NUM_TESTING_INSTANCES):
			x_test[j][i] = (x_test[j][i] - mean)/sdev
		
	return x_train, x_test

def transform(x_train, x_test):

	for i in range(NUM_TRAINING_INSTANCES):
		for j in range(NUM_FEATURES-1):
			x_train[i][j] = math.log(x_train[i][j] + 1)
	
	for i in range(NUM_TESTING_INSTANCES):
		for j in range(NUM_FEATURES-1):
			x_test[i][j] = math.log(x_test[i][j] + 1)

	return x_train, x_test

def binarize(x_train, x_test):

	for i in range(NUM_TRAINING_INSTANCES):
		for j in range(NUM_FEATURES-1):
			if(x_train[i][j] > 0):
				x_train[i][j] = 1
			else:
				x_train[i][j] = 0

	for i in range(NUM_TESTING_INSTANCES):
		for j in range(NUM_FEATURES-1):
			if(x_test[i][j] > 0):
				x_test[i][j] = 1
			else:
				x_test[i][j] = 0

	return x_train, x_test

def gradientDescent(w, x, y, a):
	print("Descending:")
	while(1):
		wPlus = w + (a * ( np.dot(   x.transpose(), y - sigmoid(np.dot(x,w)))          ))
		wPlus_loss = lossFunction(wPlus, x, y)
		#print(wPlus_loss)
		if( abs(wPlus_loss - lossFunction(w,x,y)) <= EPSILON):
			return wPlus
		w = wPlus

def gradientDescent_bin(w, x, y, a):
	print("\tDescending:")
	

	while(1):
		wPlus = w + (a * ( np.dot(   x.transpose(), y - sigmoid(np.dot(x,w)))          ))
		
		#print("\t\tcalc wPlus_loss: ")
		wPlus_loss = lossFunction(wPlus, x, y)
		#print("\t\tcalc w_loss: ")
		w_loss = lossFunction(w,x,y)

		#print(wPlus_loss, w_loss, abs(wPlus_loss - w_loss))
		if( abs(wPlus_loss - w_loss) <= EPSILON):
			print("Alpha Used: " + str(a))
			return wPlus
		
		if( abs(wPlus_loss - w_loss) < 300):
			a = a / 10
		
		w = wPlus

def gradientDescent_trans(w, x, y, a):
	print("\tDescending:")
	

	while(1):
		wPlus = w + (a * ( np.dot(   x.transpose(), y - sigmoid(np.dot(x,w)))          ))
		
		#print("\t\tcalc wPlus_loss: ")
		wPlus_loss = lossFunction(wPlus, x, y)
		#print("\t\tcalc w_loss: ")
		w_loss = lossFunction(w,x,y)

		#print(wPlus_loss, w_loss, abs(wPlus_loss - w_loss))
		if( abs(wPlus_loss - w_loss) <= EPSILON_TRANS):
			print("Alpha Used: " + str(a))
			return wPlus
		
		if( abs(wPlus_loss - w_loss) < 300):
			a = a / 10
		
		w = wPlus

def lossFunction(w, x, y):
	rng = len(y)
	loss = 0
	
	for i in range(rng):
		#print("\t\t\twTx: " + str(np.dot(w.transpose(),x[i])) + str(i))
		loss = loss + (y[i]) * math.log( sigmoid(np.dot(w.transpose(),x[i]))) + (1-y[i]) * math.log( sigmoid(np.dot(w.transpose(),x[i])) )
	return loss

def error_collector(w,x,y):
	rng = len(y)
	count = 0
	wrong = 0

	for i in range(rng):
		out = S(np.dot(w.transpose(),x[i]))
		if( out > .5 ):
			prediction = 1
		else:
			prediction = 0
		if ( prediction != int(y[i]) ):
			wrong = wrong + 1
		count = count + 1	

	return float(wrong)/float(count)


def standard():
	w = np.ones((NUM_FEATURES,1))

	x_train, y_train = getTrainingData()
	x_test, y_test = getTestingData()

	print("Creating standardized..")
	x_train_standardized, x_test_standardized = standardize(x_train, x_test)

	w_standardized = gradientDescent(w, x_train_standardized, y_train, 0.0001)

	print("a)")
	print("Training Error:")
	print(error_collector(w_standardized, x_train_standardized, y_train))
	print("Testing Error:")
	print(error_collector(w_standardized, x_test_standardized, y_test))

def trans():
	w = np.ones((NUM_FEATURES,1))

	x_train, y_train = getTrainingData()
	x_test, y_test = getTestingData()

	print("Creating transformed..")
	x_train_transformed, x_test_transformed = transform(x_train, x_test)

	w_transformed = gradientDescent_trans(w, x_train_transformed, y_train, 0.00001)

	print("b)")
	print("Training Error:")
	print(error_collector(w_transformed, x_train_transformed, y_train))
	print("Testing Error:")
	print(error_collector(w_transformed, x_test_transformed, y_test))


def bin():
	w = np.ones((NUM_FEATURES,1))

	x_train, y_train = getTrainingData()
	x_test, y_test = getTestingData()

	print("Creating binarized..")
	x_train_binarized, x_test_binarized = binarize(x_train, x_test)

	w_binarized = gradientDescent_bin(w, x_train_binarized, y_train,0.01)

	print("c)")
	print("Training Error:")
	print(error_collector(w_binarized, x_train_binarized, y_train))
	print("Testing Error:")
	print(error_collector(w_binarized, x_test_binarized, y_test))

standard()
trans()
bin()

