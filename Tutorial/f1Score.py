import numpy as np
import sklearn.metrics as metrics

a = np.array([1,1,1,1,0,0]) 
predict = [[2.1,5,3,1,0,0],[1,1,1,1,6,7]]
b = np.array([1,1,0,0,0,0]) 
b = np.array(np.argmin(predict,0), int)
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

f1s = metrics.f1_score(a,b)
print ('f1s ', f1s )