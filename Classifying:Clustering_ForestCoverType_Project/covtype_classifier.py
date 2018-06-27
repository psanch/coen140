# Pedro Sanchez

# ==================================================
#          IMPORTS
# ==================================================

import numpy as np
import math

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# ==================================================
#          DEFINE CONSTANTS
# ==================================================

NUM_DATASET_INSTANCES = 581012

NUM_FEATURES = 54 + 1

TRAIN_FILE = "covtype_training.txt"
TEST_FILE = "covtype_testing.txt"

# ==================================================
#          PARSE DATA FUNCTIONS
# ==================================================

def formatFileIntoNumpy(filename):
	
	rawData = open(filename,"r")
	lines = rawData.read().split("\n")

	data_list = []
	y_list = []

	for i in range(len(lines)):
		string = lines[i].split(",")
		y_list.append(string[NUM_FEATURES-1:])
		data_list.append(string[:-1])

	x = np.ones((len(lines),NUM_FEATURES))
	y = np.ones((len(lines),1))

	for i in range(len(lines)):
		y[i] = y_list[i]
		for j in range(NUM_FEATURES-1):
			x[i][j] = data_list[i][j]

	return x,y


def getData(train_fname, test_fname):
	x_train, y_train = formatFileIntoNumpy(train_fname)
	x_test, y_test = formatFileIntoNumpy(test_fname)
	return x_train, y_train, x_test, y_test

# ==================================================
#          HELPER FUNCTIONS
# ==================================================

def getAccuracy(a,b):
	if(len(a) != len(b)):
		print("getAccuracy needs elements of same length!")
		return -1
	num = len(a)
	correct = 0
	for i in range(num):
		if( int(a[i]) == int(b[i]) ):
			correct+=1
	acc = float(correct)/float(num)
	acc*=100
	print("\tAccuracy %:\n" + "\t" + str(acc))
	return acc

# ==================================================
#          EXECUTE
# ==================================================

print("Getting data...")
x_train, y_train, x_test, y_test = getData(TRAIN_FILE, TEST_FILE)

print("Training LDA...")
lda = LinearDiscriminantAnalysis(solver="svd")
model = lda.fit(x_train, y_train.ravel())

print("Predicting LDA over Training Data...")
y_train_pred = model.predict(x_train)
LDA_train_accuracy = getAccuracy(y_train_pred, y_train)

print("Predicting LDA over Testing Data...")
y_test_pred = model.predict(x_test)
LDA_test_accuracy = getAccuracy(y_test_pred, y_test)


print("Training QDA...")
qda = QuadraticDiscriminantAnalysis()
model = qda.fit(x_train, y_train.ravel())

print("Predicting QDA over Training Data...")
y_train_pred = model.predict(x_train)
QDA_train_accuracy = getAccuracy(y_train_pred, y_train)

print("Predicting QDA over Testing Data...")
y_test_pred = model.predict(x_test)
QDA_test_accuracy = getAccuracy(y_test_pred, y_test)






