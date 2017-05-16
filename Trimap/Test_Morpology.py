#import tensorflow as tf
import numpy as np
import scipy
from scipy import ndimage
import matplotlib.pyplot as plt

#x_data = np.float32(np.random.rand(5, 5))
#print (x_data)
h = 6
w = 6
allOne = a = np.ones([h,w])
a = np.ones([h,w])
for y in range(h):
    for x in range(w):
        a[y,x] = 0

for y in range(h):
    for x in range(w):
        if np.abs(h/2.0-y) < 2:
            if np.abs(w/2.0 - x) < 2:
                a[y,x] = 1

def ConvertToTrimap(src):
    h = src.shape[0]
    w = src.shape[1]
    kernel = (3,3)
    allOne = np.ones([h,w])
    dilat = ndimage.grey_dilation(src, size=kernel)        
    foreground = ndimage.grey_erosion(src, size=kernel)        
    unknown = dilat-foreground
    tripmap =  unknown * 0.5 + foreground
    return tripmap

print (a)
print ('')
dilat = ndimage.grey_dilation(a, size=(3,3))
print (dilat)
print ('foreground')
foreground = ndimage.grey_erosion(a, size=(3,3))
print (foreground)
print ('unknown')
unknown = dilat-foreground
print (unknown )
print ('background')
print (allOne-dilat)

tripmap =  unknown * 0.5 + foreground
print ('tripmap ')
print (ConvertToTrimap(a) )