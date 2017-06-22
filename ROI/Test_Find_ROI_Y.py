#import tensorflow as tf
import numpy as np

def GetNonZeroY(src):
    print ('src', np.shape(src))
    src_col_sum = np.sum(src, axis=1)
    print ('src_col_sum', np.shape(src_col_sum))
    print (src_col_sum)
    
    y0 = 0
    y1 = len(src_col_sum)
    for y in range(len(src_col_sum)):
        if src_col_sum[y]>0:
            y0 = y
            break
    for y in range(y1):
        idx = y1-y-1
        if src_col_sum[idx]>0:
            y1 = idx
            break

    print ('height: %d, y0~y1 = %d ~ %d' %(len(src_col_sum), y0,y1))
    return y0,y1

x = [[0,0,0,0,0], 
     [0,1,1,0,0], 
     [1,1,1,1,1], 
     [0,0,1,1,1], 
     [0,0,0,0,0]] 

x2 = [[0,1,0,0,0], 
     [0,1,1,0,0], 
     [1,1,1,1,1], 
     [0,0,0,0,0], 
     [0,0,0,0,0]] 

x = np.array(x)
x2 = np.array(x2)

y0,y1 = GetNonZeroY(x)
print (x[y0:y1+1,:])

y0,y1 = GetNonZeroY(x2)
print (x2[y0:y1+1,:])


