import numpy as np

l = np.arange(12)
l = np.reshape(l, [2,6])
even = l[:, ::2]         # even  - start at the beginning at take every second item
[0, 2, 4, 6, 8]
odd = l[:,1::2]    

print (l.shape, l)
print (even .shape, even )
print (odd.shape, odd)