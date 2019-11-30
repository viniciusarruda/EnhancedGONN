import numpy as np

filepath = 'iris.data'

with open(filepath) as f:
	data = f.readlines()

data = [d.strip().split(',') for d in data]
data = [['0' if i == 'Iris-setosa' else i for i in l] for l in data]
data = [['1' if i == 'Iris-versicolor' else i for i in l] for l in data]
data = [['2' if i == 'Iris-virginica' else i for i in l] for l in data]

data = np.array(data, dtype=np.float)

Y = data[:, -1][:, np.newaxis]
X = data[:, :-1]

data = np.concatenate((X, Y), axis=1)

print(data.shape)

np.save('iris.npy', data)
