import matplotlib.image as mp_image
import matplotlib.pyplot as plt

def func(input):
    a = input+1
    input = 100
    return a,input

x = 3
print (func(x))
print ('x', x)

ex = [1, ["a","b"], ("tuple0","tuple1")]
print ('len',len(ex))
print (ex[1])

mylist = ["A",2,3.5]
print ('len',len(mylist ))
print (mylist[1])

mytuple = (1,2,3)
print ('len',len(mytuple ))
print (mytuple[1])

filename = "C:/Users/pc/Pictures/ship.jpg"
image = mp_image.imread(filename)
print ('input dim = {}'.format(image.ndim))
print ('input shape = {}'.format(image.shape))

import os
dir = os.path.dirname(os.path.realpath(__file__))
print ('dir', dir)
#plt.imshow(image)
#plt.show()