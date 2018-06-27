import numpy as np
import csv
import math

NUM_TRAINING_FEATURES = 96
NUM_TRAINING_INSTANCES = 1595

NUM_TESTING_FEATURES = 96
NUM_TESTING_INSTANCES = 399

LAMBDA_VALUES = [400,200,100,50,25,12.5,6.25,3.125,1.5625,0.78125]
NUM_LAMBDA_VALUES = 10

EPSILON = 0.00001
ALPHA = 0.00005

def trainLR(m,y):
	
	return np.dot( np.linalg.inv(np.dot(m.transpose(),m)) , (np.dot(m.transpose(),y)))

def trainRR(m,y,l,num_instances):
	lam = np.zeros((num_instances,num_instances))
	lam = lam + l
	lam = np.diag(np.diag(lam))
	return np.dot( np.linalg.inv(np.dot(m.transpose(),m) + lam ) , (np.dot(m.transpose(),y)))

	data = open("crime-train.txt", "r")
	lines = data.read().split("\n")
	data.close()

	lines = lines[1:-1]
	temp_list = []
	y_list = []

	for i in range(NUM_TRAINING_INSTANCES):
		string = lines[i].split(",")
		y_list.append(string[:-(NUM_TRAINING_FEATURES-1)])
		temp_list.append(string[1:])

	m = np.ones((NUM_TRAINING_INSTANCES,NUM_TRAINING_FEATURES))
	y = np.ones((NUM_TRAINING_INSTANCES,1))

	for i in range(NUM_TRAINING_INSTANCES):
		y[i] = y_list[i]
		for j in range(NUM_TRAINING_FEATURES-1):
			m[i][j] = temp_list[i][j]

	return m,y

def formatTestingData():

	data = open("crime-test.txt", "r")
	lines = data.read().split("\n")
	data.close()

	lines = lines[1:-1]
	temp_list = []
	y_list = []

	for i in range(NUM_TESTING_INSTANCES):
		string = lines[i].split(",")
		y_list.append(string[:-(NUM_TESTING_FEATURES-1)])
		temp_list.append(string[1:])

	m = np.ones((NUM_TESTING_INSTANCES,NUM_TESTING_FEATURES))
	y = np.ones((NUM_TESTING_INSTANCES,1))

	for i in range(NUM_TESTING_INSTANCES):
		y[i] = y_list[i]
		for j in range(NUM_TESTING_FEATURES-1):
			m[i][j] = temp_list[i][j]

	return m,y

def RMSE(yNew, yTrue, nTest):
	mysum = 0.0
	for i in range(nTest):
		mysum = mysum + ((yNew[i] - yTrue[i])**2)
	mysum = mysum / nTest
	return math.sqrt(mysum)

def MSE_range(yNew, yTrue, start, end):
	mysum = 0.0
	for i in range(start,end):
		mysum = mysum + ((yNew[i] - yTrue[i])**2)
	mysum = mysum / (end-start)
	return mysum

def predict(w, x, num_instances):
	yNew = np.ones((num_instances,1))
	for i in range(num_instances):
		yNew[i] = np.dot(w.transpose(),x[i])
	return yNew

def lossFunction(w, x, y):
	a = (np.dot(x,w)-y)
	return np.dot(a.transpose(),a)

def GD(w, x, y):

	wPlus = w + (ALPHA)*np.dot(x.transpose(),(y-np.dot(x,w)))
	while( abs((lossFunction(wPlus,x,y)) - (lossFunction(w,x,y))) >= EPSILON ):
		w = wPlus
		wPlus = w + (ALPHA)*np.dot(x.transpose(),y-np.dot(x,w))
	return wPlus

def CV5(l, x, y):
	
	k = 5
	step = 319
	mysum = 0

	for i in range(k):
		
		x_test = x[i*step:(i+1)*step]
		y_test = y[i*step:(i+1)*step]
		
		x_train = np.delete(x, np.s_[step*i:step*(i+1)], axis=0)
		y_train = np.delete(y, np.s_[step*i:step*(i+1)], axis=0)
		
		#RR Training
		lam = np.zeros((96,96))
		lam = lam + l
		lam = np.diag(np.diag(lam))

		w = np.dot(np.linalg.inv(np.dot(x_train.transpose(),x_train) + lam) , (np.dot(x_train.transpose(),y_train)))
		y_train_new = predict(w, x_train, 1276)

		mysum = mysum + RMSE(y_train_new, y_train, 1276)

	return mysum / k

def getOptimalLambdaClosedRR(x_training, y_training_true):
	#print("\ttesting lambda[9]")
	CV5_error = CV5(LAMBDA_VALUES[9], x_training, y_training_true)
	lambda_index = 9
	#print("\terror for lambda[9] = "+str(CV5_error))

	for i in range(9):
		#print("\ttesting lambda["+str(i)+"] = "+str(LAMBDA_VALUES[i]))
		temp_error = CV5(LAMBDA_VALUES[i],x_training,y_training_true)
		#print("\terror for lambda["+str(i)+"] = "+str(temp_error))
		if( CV5_error > temp_error):
			CV5_error = temp_error
			lambda_index = i
	return lambda_index

def GDL(w, x, y, l):
	wPlus = w + (ALPHA)*(np.dot(x.transpose(),y-np.dot(x,w)) - l*w)
	while( abs((lossFunction(wPlus,x,y)) - (lossFunction(w,x,y))) >= EPSILON ):
		w = wPlus
		wPlus = w + (ALPHA)*(np.dot(x.transpose(),y-np.dot(x,w)) - l*w)
	return wPlus

def CV5GD(l, x, y):
	
	k = 5
	step = 319
	mysum = 0

	for i in range(k):
		
		x_test = x[i*step:(i+1)*step]
		y_test = y[i*step:(i+1)*step]
		
		x_train = np.delete(x, np.s_[step*i:step*(i+1)], axis=0)
		y_train = np.delete(y, np.s_[step*i:step*(i+1)], axis=0)
		
		#RR Training
		lam = np.zeros((96,96))
		lam = lam + l
		lam = np.diag(np.diag(lam))
		w_testing_LR_GD = np.zeros((96,1))

		for i in range(NUM_TESTING_FEATURES):
			w_testing_LR_GD[i] = np.random.normal(0,1)

		w = GDL(w_testing_LR_GD, x, y, l)

		y_train_new = predict(w, x_train, 1276)

		mysum = mysum + RMSE(y_train_new, y_train, 1276)

	return mysum / k

def getOptimalLambdaGD(x_training, y_training_true):
	#print("\ttesting lambda[9] = " + str(LAMBDA_VALUES[9]))
	CV5_error = CV5GD(LAMBDA_VALUES[9], x_training, y_training_true)
	lambda_index = 9
	#print("\terror for lambda[9] = "+str(CV5_error))
	for i in range(9):
		#print("\ttesting lambda["+str(i)+"] = " + str(LAMBDA_VALUES[i]))
		temp_error = CV5GD(LAMBDA_VALUES[i],x_training,y_training_true)
		#print("\terror for lambda["+str(i)+"] = "+str(temp_error))

		if( CV5_error > temp_error):
			CV5_error = temp_error
			lambda_index = i

	return lambda_index

def main():

	print("#1:")
	x_training, y_training_true = formatTrainingData()
	w = trainLR(x_training, y_training_true)
	y_training_new = predict(w, x_training, NUM_TRAINING_INSTANCES)
	error_training = RMSE(y_training_new, y_training_true, NUM_TRAINING_INSTANCES)

	print("Training Error:")
	print(error_training)

	x_testing, y_testing_true = formatTestingData()
	y_testing_new = predict(w, x_testing, NUM_TESTING_INSTANCES)
	error_testing = RMSE(y_testing_new, y_testing_true, NUM_TESTING_INSTANCES)

	print("Testing Error:")
	print(error_testing)

#---------

	print("#2:")
	#find lambda for closed solution in RR

	index = getOptimalLambdaClosedRR(x_training, y_training_true)
	optimal_lambda = LAMBDA_VALUES[index]
	print("optimal lambda using closed form RR: " + str(optimal_lambda))

	#RMSE on test data using optimal lambda in closed solution

	w = trainRR(x_training, y_training_true, optimal_lambda, NUM_TRAINING_FEATURES)
	y_testing_new = predict(w, x_testing, NUM_TESTING_INSTANCES)
	error_RR_closed = RMSE(y_testing_new, y_testing_true, NUM_TESTING_INSTANCES)

	print("error closed form RR = " + str(error_RR_closed))

#---------
	
	print("#3:")
	#LR using GD, get RSME for test/train data
		#training data
	w_training_LR_GD = np.zeros((NUM_TRAINING_FEATURES,1))
	for i in range(NUM_TRAINING_FEATURES):
		w_training_LR_GD[i] = np.random.normal(0,1)
		
	w_training_LR_GD = GD(w_training_LR_GD, x_training, y_training_true)
	y_training_new_LR_GD = predict(w_training_LR_GD, x_training, NUM_TRAINING_INSTANCES)
	error_training_LR_GD = RMSE(y_training_new_LR_GD, y_training_true, NUM_TRAINING_INSTANCES)
	print("Training Error:")
	print(error_training_LR_GD)
	
		#testing data
	w_testing_LR_GD = np.zeros((NUM_TESTING_FEATURES,1))
	for i in range(NUM_TESTING_FEATURES):
		w_testing_LR_GD[i] = np.random.normal(0,1)
	w_testing_LR_GD = GD(w_testing_LR_GD, x_testing, y_testing_true)
	y_testing_new_LR_GD = predict(w_testing_LR_GD, x_testing, NUM_TESTING_INSTANCES)
	error_testing_LR_GD = RMSE(y_testing_new_LR_GD, y_testing_true, NUM_TESTING_INSTANCES)
	print("Testing Error:")
	print(error_testing_LR_GD)

#---------

	print("#4:")

	#RR using GD
	#find lambda

	index_GD = getOptimalLambdaGD(x_training, y_training_true)

	optimal_lambda_GD = LAMBDA_VALUES[index_GD]
	print("Lambda = "+str(LAMBDA_VALUES[index_GD]))

	#train w using lambda
	w_testing_RR_GD = np.zeros((NUM_TESTING_FEATURES,1))
	for i in range(NUM_TESTING_FEATURES):
		w_testing_RR_GD[i] = np.random.normal(0,1)
	w_testing_RR_GD = GDL(w_testing_RR_GD, x_testing, y_testing_true, optimal_lambda_GD)
	
	#predict on testing data
	y_testing_new_RR_GD = predict(w_testing_RR_GD, x_testing, NUM_TESTING_INSTANCES)

	#compare error
	error_testing_RR_GD = RMSE(y_testing_new_RR_GD, y_testing_true, NUM_TESTING_INSTANCES)
	print("Testing Error:")
	print(error_testing_RR_GD)


main()
