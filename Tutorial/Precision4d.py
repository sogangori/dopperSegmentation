import numpy as np

a = np.array([[1,0],[1,0],[0,1]])
b = np.array([[1,0],[0,1],[0,1]])

a1 = np.reshape(a,[-1])

print ('len',a1.shape, len(a1))
print ('len',a.shape, len(a))
a = np.argmax(a,1)
b = np.argmax(b,1)
print ('len',a.shape, len(a))
print (a)
print (b)



accuracy = np.mean(np.equal(a,b))
right = np.sum(b * 2 - a == 1)
precision = right / np.sum(b)
recall = right / np.sum(a)
f1 = 2 * precision*recall/(precision+recall)

print ('accuracy',accuracy)
print ('precision', precision)
print ('recall', recall)
print ('f1', f1)