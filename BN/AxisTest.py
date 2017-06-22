import numpy as np

x = np.array([[1,10],[2,20],[3,30],[4,40]])

print (x.shape)
print (x)

print ( np.sum(x,axis=0))
print ( np.sum(x,axis=1))

x = np.array([[[1,10,100],[2,20,200]],[[3,30,300],[4,40,400]]])

print (x.shape)
print (x)

print ('axis=0', np.sum(x,axis=0))
print ('axis=1', np.sum(x,axis=1))
print ('axis=2', np.sum(x,axis=2))
#print ('axis=[0,1]', np.sum(x,axis=[0,1]))
#print ('axis=[0,1,2]', np.sum(x,axis=[0,1,2]))

print ( np.reshape(x, [-1]))