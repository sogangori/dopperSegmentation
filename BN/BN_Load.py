import tensorflow as tf
import numpy as np

x = np.array([[1,10],[2,20],[3,30],[4,40]])
y = np.array([[21],[42],[63],[84]])
import tensorflow as tf
import numpy as np
import pickle
import BN

sess = tf.Session()
saver = tf.train.Saver()  
    
saver.restore(sess, './bn_weight')
print("Model restored")

filters = []
for var in tf.trainable_variables(): 
    print (var)
    filter = sess.run(var)
    filters.append(filter)
    print (filter)
print ('filters', len(filters))
