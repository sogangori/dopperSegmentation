﻿import tensorflow as tf
import numpy as np
import Model_bimap_small as model 
import pickle

sess = tf.Session()
saver = tf.train.Saver()  
    
saver.restore(sess, model.modelName)
print("Model restored")

filters = []
for var in tf.trainable_variables(): 
    print (var)
    filter = sess.run(var)
    filters.append(filter)
    print (filter.shape)
print ('filters', len(filters))
name_of_file = 'variable_small.txt'
file = open(name_of_file ,'w')

#f = open(name_of_file,'wb')
filtersElement = []
for filter in filters:
    if np.ndim(filter) >1:
        W = filter.shape[0]
        H = filter.shape[1]
        C = filter.shape[2]
        N = filter.shape[3]
        print ('np', H,W,C,N)
        #HWCN -> NCHW
        #for y in range(H):
        #    for x in range(W):
        #        for 
        #index_in = 
    arr = np.reshape(filter, [-1])
    filtersElement.extend(arr)

print ('filtersElement', len(filtersElement))
data = np.array(filtersElement)
print ('data', data.shape)
index = 0
data.tofile(file,'\n','%.4f')
file.close()

#with file('variable.txt', 'w') as outfile:
#    outfile.write(filters)