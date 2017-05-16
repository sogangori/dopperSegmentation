import numpy as np 

arr = np.array( [[[1,2,2],[3,4,4]],[[5,6,7],[7,8,8]]])
re= arr[::-1]
b = np.rot90(arr)
c = np.rot90(b)
d = np.fliplr(arr)

print (arr.shape, arr)
print (b.shape, b)
print (c.shape, c)
print (d.shape, d)