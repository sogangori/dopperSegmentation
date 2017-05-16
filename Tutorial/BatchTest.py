import numpy as np

size = 10
EVAL_BATCH_SIZE = 4
for begin in range(0, size, EVAL_BATCH_SIZE):
    end = begin + EVAL_BATCH_SIZE
    
    end = np.minimum(size, end)-1
    print ( begin, end)