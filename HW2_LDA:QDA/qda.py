import numpy as np
import math

def gaussianTest(xi, mu, sigma):
	const = (1)/(math.sqrt(np.linalg.det(sigma)))
	item = (xi-mu)
	power = (-1/2)*( item )*(np.linalg.inv(sigma))*( item.transpose() )
	return (const* math.exp(power))

print("QDA:")

#DEFINE constants
dataset_rows = 150
category_rows = 50
category_testing_rows = 10
category_training_rows = 40
category_columns = 4
range_features = 4


#FILE I/O -------------

data = open("dataset_iris.txt", "r")
lines = data.read().split("\n")
data.close()

num_data = len(lines) #150

dataset_iris = [[float(0) for x in range(range_features)] for y in range(num_data)]

##END FILE I/O --------







#POPULATE Lists of Matrices -----------
#SETOSA

list_setosa_training = []

for i in range(40):
    dataset_iris[i] = lines[i].split(",")
    dataset_iris[i] = dataset_iris[i][:-1]
    dataset_iris_string = ','.join(dataset_iris[i])
    a = np.matrix(dataset_iris_string)
    list_setosa_training.append(a)

list_setosa_testing = []

for i in range(40,50):
    dataset_iris[i] = lines[i].split(",")
    dataset_iris[i] = dataset_iris[i][:-1]
    dataset_iris_string = ','.join(dataset_iris[i])
    a = np.matrix(dataset_iris_string)
    list_setosa_testing.append(a)

#VERSICOLOR

list_versicolor_training = []

for i in range(50,90):
    dataset_iris[i] = lines[i].split(",")
    dataset_iris[i] = dataset_iris[i][:-1]
    dataset_iris_string = ','.join(dataset_iris[i])
    a = np.matrix(dataset_iris_string)
    list_versicolor_training.append(a)

list_versicolor_testing = []

for i in range(90,100):
    dataset_iris[i] = lines[i].split(",")
    dataset_iris[i] = dataset_iris[i][:-1]
    dataset_iris_string = ','.join(dataset_iris[i])
    a = np.matrix(dataset_iris_string)
    list_versicolor_testing.append(a)

#VIRGINICA

list_virginica_training = []

for i in range(100,140):
    dataset_iris[i] = lines[i].split(",")
    dataset_iris[i] = dataset_iris[i][:-1]
    dataset_iris_string = ','.join(dataset_iris[i])
    a = np.matrix(dataset_iris_string)
    list_virginica_training.append(a)

list_virginica_testing = []

for i in range(140,150):
    dataset_iris[i] = lines[i].split(",")
    dataset_iris[i] = dataset_iris[i][:-1]
    dataset_iris_string = ','.join(dataset_iris[i])
    a = np.matrix(dataset_iris_string)
    list_virginica_testing.append(a)

#END# POPULATE Lists of Matrices -----------


#CALCULATE mu's
setosa_mu = np.zeros(4)
versicolor_mu = np.zeros(4)
virginica_mu = np.zeros(4)

for i in range(40):
	setosa_mu = setosa_mu + list_setosa_training[i]
setosa_mu = setosa_mu / category_training_rows

for i in range(40):
	versicolor_mu = versicolor_mu + list_versicolor_training[i]
versicolor_mu = versicolor_mu / category_training_rows

for i in range(40):
	virginica_mu = virginica_mu + list_virginica_training[i]
virginica_mu = virginica_mu / category_training_rows

#print(setosa_mu)
#print(versicolor_mu)
#print(virginica_mu)

#END CALCULATE MUs

#CALCULATE COVARIANCEs

setosa_covariance = np.zeros((4,4))
versicolor_covariance = np.zeros((4,4))
virginica_covariance = np.zeros((4,4))

for i in range(40):
	item = (list_setosa_training[i] - setosa_mu)
	setosa_covariance = setosa_covariance + ( item.transpose() * item )
setosa_covariance = setosa_covariance / 40

for i in range(40):
	item = (list_versicolor_training[i] - versicolor_mu)
	versicolor_covariance = versicolor_covariance + ( item.transpose() * item )
versicolor_covariance = versicolor_covariance / 40

for i in range(40):
	item = (list_virginica_training[i] - virginica_mu)
	virginica_covariance = virginica_covariance + ( item.transpose() * item )
virginica_covariance = virginica_covariance / 40


#print(setosa_covariance)
#print(versicolor_covariance)
#print(virginica_covariance)

#END CALCULATE COVARIANCES

	


#TEST TESTING DATA:

num_test_incorrect = 0.0
num_test_total = 0.0
test_error = 0.0

for i in range(10):
	p_setosa = gaussianTest(list_setosa_testing[i], setosa_mu, setosa_covariance)
	p_versicolor = gaussianTest(list_setosa_testing[i], versicolor_mu, versicolor_covariance)
	p_virginica = gaussianTest(list_setosa_testing[i], virginica_mu, virginica_covariance)
	
	prediction = max(p_setosa,p_versicolor,p_virginica)
	if(prediction != p_setosa): num_test_incorrect = num_test_incorrect + 1
	num_test_total = num_test_total + 1
	
for i in range(10):
	p_setosa = gaussianTest(list_versicolor_testing[i], setosa_mu, setosa_covariance)
	p_versicolor = gaussianTest(list_versicolor_testing[i], versicolor_mu, versicolor_covariance)
	p_virginica = gaussianTest(list_versicolor_testing[i], virginica_mu, virginica_covariance)
	
	prediction = max(p_setosa,p_versicolor,p_virginica)
	if(prediction != p_versicolor): num_test_incorrect = num_test_incorrect + 1
	num_test_total = num_test_total + 1
	
for i in range(10):
	p_setosa = gaussianTest(list_virginica_testing[i], setosa_mu, setosa_covariance)
	p_versicolor = gaussianTest(list_virginica_testing[i], versicolor_mu, versicolor_covariance)
	p_virginica = gaussianTest(list_virginica_testing[i], virginica_mu, virginica_covariance)
	
	prediction = max(p_setosa,p_versicolor,p_virginica)
	if(prediction != p_virginica): num_test_incorrect = num_test_incorrect + 1
	num_test_total = num_test_total + 1

test_error = num_test_incorrect/num_test_total
print("Testing error:")
print(test_error)

#END TEST TESTING DATA



#TEST TRAINING DATA

num_train_incorrect = 0.0
num_train_total = 0.0
train_error = 0.0

for i in range(40):
	p_setosa = gaussianTest(list_setosa_training[i], setosa_mu, setosa_covariance)
	p_versicolor = gaussianTest(list_setosa_training[i], versicolor_mu, versicolor_covariance)
	p_virginica = gaussianTest(list_setosa_training[i], virginica_mu, virginica_covariance)
	
	prediction = max(p_setosa,p_versicolor,p_virginica)
	if(prediction != p_setosa): num_train_incorrect = num_train_incorrect + 1
	num_train_total = num_train_total + 1
	
for i in range(40):
	p_setosa = gaussianTest(list_versicolor_training[i], setosa_mu, setosa_covariance)
	p_versicolor = gaussianTest(list_versicolor_training[i], versicolor_mu, versicolor_covariance)
	p_virginica = gaussianTest(list_versicolor_training[i], virginica_mu, virginica_covariance)
	
	prediction = max(p_setosa,p_versicolor,p_virginica)
	if(prediction != p_versicolor): num_train_incorrect = num_train_incorrect + 1
	num_train_total = num_train_total + 1
	
for i in range(40):
	p_setosa = gaussianTest(list_virginica_training[i], setosa_mu, setosa_covariance)
	p_versicolor = gaussianTest(list_virginica_training[i], versicolor_mu, versicolor_covariance)
	p_virginica = gaussianTest(list_virginica_training[i], virginica_mu, virginica_covariance)
	
	prediction = max(p_setosa,p_versicolor,p_virginica)
	if(prediction != p_virginica): num_train_incorrect = num_train_incorrect + 1
	num_train_total = num_train_total + 1

train_error = num_train_incorrect/num_train_total
print("Training error:")
print(train_error)


#END TEST TRAINING DATA
