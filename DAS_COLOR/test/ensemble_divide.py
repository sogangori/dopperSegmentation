import numpy as np
import tensorflow as tf

ensemble = 3
dataSize = 10

a = [0,1,2,3,4,5,6,7,8,9]
print ('a',a)
print ('a[0:ensemble]',a[0:ensemble])
for step in range(0,20):
    ensemble_start = step % (dataSize-ensemble+1)
    ensemble_end = ensemble_start + ensemble
    ensemble_src = a[ensemble_start:ensemble_end]
    print ('step', step,ensemble_start,ensemble_end,ensemble_src)