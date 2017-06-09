import numpy as np

a = np.array([1,1,1,1,0,0])
b = np.array([1,1,0,0,0,0])

accuracy = np.mean(np.equal(a,b))
right = np.sum(b * a == 1)
precision = right / np.sum(b)
recall = right / np.sum(a)
f1 = 2 * precision*recall/(precision+recall)

print ('a*b', a*b)
print ('accuracy',accuracy)
print ('precision', precision)
print ('recall', recall)
print ('f1', f1)