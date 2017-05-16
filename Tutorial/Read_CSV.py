import numpy as np

data = np.loadtxt('./data.csv', delimiter=',', dtype=np.float32)
x = data[:,0:-1]
y = data[:,-1]
print ('x',x)
print ('y',y)