import numpy as np

a = np.array(range(10))
b = np.reshape(a, [2,5])
c = [10,100]
c = np.reshape(c, [2,1])

d = np.multiply(b , c)

print ('a',a)
print ('b',b)
print ('c',c)
print ('d',d)