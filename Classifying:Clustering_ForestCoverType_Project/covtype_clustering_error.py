# Pedro Sanchez

# ==================================================
#          IMPORTS
# ==================================================

import numpy as np
import math

from sklearn.cluster import MiniBatchKMeans, KMeans
from sklearn.metrics.pairwise import euclidean_distances

# ==================================================
#          DEFINE CONSTANTS
# ==================================================

NUM_DATASET_INSTANCES = 581012
NUM_TRAINING_INSTANCES = 15120
NUM_TESTING_INSTANCES = 565892

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

def getClusterToLabelMapping(cxl,n):
	
	hi, hiIndex = 0,0
	label_mappings = [[],[],[],[],[],[],[]]

	for i in range(n):
		hi = cxl[i][0]
		for j in range(1,n):
			if(hi < cxl[i][j]):
				hi = cxl[i][j]
		for j in range(n):
			if( hi == cxl[i][j] ):
				label_mappings[i].append(j)

	return label_mappings

# ==================================================
#          EXECUTE
# ==================================================

print("...")
x_train, y_train, x_test, y_test = getData(TRAIN_FILE, TEST_FILE)

num_clusters = 2

while( num_clusters <= 3 ):
	#fit k_means over training data
	k_means = KMeans(n_clusters=num_clusters, init='k-means++', n_init=10).fit(x_train)

	centroids = np.ones((num_clusters, NUM_FEATURES))
	centroids = k_means.cluster_centers_

	print(centroids)

	training_error = 0.0
	testing_error = 0.0

	#get cluster predictions based on fit over training data
	clusters_train = k_means.predict(x_train)



	for i in range(len(clusters_train)):
		print("x_train[i]")
		print(x_train[i])
		print("centroids[clusters_train[i]")
		print(centroids[clusters_train[i]])
		print("(x_train[i]-centroids[clusters_train[i]])")
		print((x_train[i]-centroids[clusters_train[i]]))
		print("(x_train[i]-centroids[clusters_train[i]]).transpose()")
		print((x_train[i]-centroids[clusters_train[i]]).transpose())
		print("np.dot((x_test[i]-centroids[clusters_test[i]]),(x_test[i]-centroids[clusters_test[i]]).transpose())")
		print(np.dot((x_train[i]-centroids[clusters_train[i]]),(x_train[i]-centroids[clusters_train[i]]).transpose()))
		print("\n\n\n\n\n\n")
		training_error += np.dot((x_train[i]-centroids[clusters_train[i]]),(x_train[i]-centroids[clusters_train[i]]).transpose())
	training_error /= NUM_TRAINING_INSTANCES
	
	clusters_test = k_means.predict(x_test)
	for i in range(len(clusters_test)):
		testing_error += np.dot((x_test[i]-centroids[clusters_test[i]]),(x_test[i]-centroids[clusters_test[i]]).transpose())
	testing_error /= NUM_TESTING_INSTANCES
	
	print(str(num_clusters)+",\t"+str(training_error)+",\t"+str(testing_error))

	num_clusters+=1










