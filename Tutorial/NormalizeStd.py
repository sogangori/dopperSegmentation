import numpy as np
from sklearn.preprocessing import StandardScaler

data = np.random.randint(10, size=(4,2))
print ('rank',np.rank(data))
print ('shape',np.shape(data))
print ('rank0',np.shape(data)[0])
print ('rank1',np.shape(data)[1])
print ('rank-1',np.shape(data)[-1])
print (data)

d2 = np.reshape(data,(2,2,-1))
print ('reshape',d2)
standardized_data = StandardScaler().fit_transform(data)
print (standardized_data)