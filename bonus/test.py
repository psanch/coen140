import numpy as np

a = np.ones((4,4))


for i in range(4):
	a[i][i] = i+4
	a[i][0] = 2

a[0][0] = 200

print(a.transpose()[0])

mean = np.mean(a.transpose()[0])
sdev = np.std(a.transpose()[0])
print(mean,sdev)
print(a.transpose()[0][0])

for i in range(4):
	a.transpose()[0][i] = (a.transpose()[0][i] - mean) / sdev

print(a.transpose()[0])






