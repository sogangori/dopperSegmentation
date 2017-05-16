import numpy as np

a = np.array(range(2*2*3))
b = np.reshape(a,[2,2,3])
b2 = np.reshape(a,[2*2,3])
print (b)
c =np.max(b2, axis=0)
print ('c',c.shape)
print ( c)