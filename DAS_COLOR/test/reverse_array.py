import numpy as np

a = np.arange(16)
a = np.reshape(a, (2,2,2,2))



#b = np.flipud(np.array(a.split(),float))
b = a[::-1]
print (a)
print ('b')
print (b)