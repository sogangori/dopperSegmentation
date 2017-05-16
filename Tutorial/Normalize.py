import numpy as np

a= np.array(range(10))+1
print ('a',a)

b = a.astype(float)*0.01
print ('b',b)

st = np.std(b)
print ('st', st)

d = b / np.std(b) 
print ('d',d)

#print (a *( a < 5) + a *( a > 5 * np.mean(a)))
print (a>5)
print ((a>5) * 6)

w = (a>5) *a + np.ones(len(a))
print (w)
print (a/w)

logAlpha = 50.0
c =( 40.0 * np.log10(b)+logAlpha )/logAlpha 
#print ('c',c)

x = np.array(range(10))-5.0
print ('x',x)
x1=(x > 0) 
x2 = x*x1/ np.max(x)
print ( x2)
print ( x-x2*2)

xn= (x - np.mean(x))/np.std(x)
print ('basic normal', xn, )
print ('normal mean',np.std(xn))