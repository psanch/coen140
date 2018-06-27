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

print("\n\n...")
x_train, y_train, x_test, y_test = getData(TRAIN_FILE, TEST_FILE)

#fit k_means over training data
k_means = KMeans(n_clusters=7, init='k-means++', n_init=10).fit(x_train)

#get cluster predictions based on fit over training data
clusters_train = k_means.predict(x_train)
clusters_test = k_means.predict(x_test)

#initialize arrays to count how many times a given label/clusters is read/predicted
label_count_train = [0,0,0,0,0,0,0]
label_count_test = [0,0,0,0,0,0,0]
cluster_count_train = [0,0,0,0,0,0,0]
cluster_count_test = [0,0,0,0,0,0,0]

#initalize arrays to keep track of how many times a given label went into a given cluster
count_cluster_labels_train = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
count_cluster_labels_test = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

#populate the arrays above
for i in range(len(y_train)):
	label_count_train[int(y_train[i])-1]+=1
	cluster_count_train[int(clusters_train[i])]+=1
	count_cluster_labels_train[int(y_train[i]-1)][int(clusters_train[i])]+=1

#figure out which label is most popular within a given cluster
label_mappings_train = getClusterToLabelMapping(count_cluster_labels_train, 7)

#populate the arrays above
for i in range(len(y_test)):
	label_count_test[int(y_test[i])-1]+=1
	cluster_count_test[int(clusters_test[i])]+=1
	count_cluster_labels_test[int(y_test[i])-1][int(clusters_test[i])]+=1

#figure out which label is most popular within a given cluster
label_mappings_test = getClusterToLabelMapping(count_cluster_labels_test, 7)

print("\n# Training labels")
print(label_count_train)
print("# Training clusters")
print(cluster_count_train)

for i in range(7):
	print("\nCluster #" + str(i)+ " likely maps to label #" + str(label_mappings_train[i]))
	print(count_cluster_labels_train[i])


print("\n\n# Testing labels")
print(label_count_test)
print("# Testing clusters")
print(cluster_count_test)
for i in range(7):
	print("\nCluster #" + str(i)+ " likely maps to label #" + str(label_mappings_train[i]))
	print(count_cluster_labels_train[i])









