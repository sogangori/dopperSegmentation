
import numpy as np
import tensorflow as tf

pool_stride2 =[1, 1, 2, 1]
a = np.array( range(2*2*2),dtype=np.float32)*2
b_0 = tf.reshape(a, shape=[2,2,2])
b = tf.reshape(a, shape=[1,2,2,2])
c = tf.nn.max_pool(b, ksize=pool_stride2, strides=pool_stride2, padding="SAME")
max_out,pos = tf.nn.max_pool_with_argmax(b, ksize=pool_stride2, strides=pool_stride2, padding="SAME")
sess = tf.Session()
print ('a',a)
print ('b',b)
print (sess.run(b_0))
print ('pool')
print (sess.run(c))
print ('max_pool_with_argmax')
print ('max_out',max_out)
print ('pos ',pos)
print (sess.run(max_out))
print (sess.run(pos))
print (sess.run(b_0))
pos_index = tf.reshape(pos,[-1])

oneHot_1d = tf.sparse_to_dense(sparse_indices= pos_index,output_shape= [len(a)] ,sparse_values=1, default_value= 0)
oneHot_2d = tf.reshape(oneHot_1d, [2,2,2])
oneHot_2d = tf.cast(oneHot_2d, tf.float32)
unpool = tf.multiply(b_0 , oneHot_2d)
print (sess.run(oneHot_1d ))
print (sess.run(oneHot_2d ))
print (sess.run(unpool ))
