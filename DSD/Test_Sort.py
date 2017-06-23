import numpy as np
import tensorflow as tf

len_w = 9
#w = tf.Variable(tf.constant(0.0, shape=[depth0*2]))
w = tf.Variable(tf.range(len_w))
w = tf.cast(w, tf.float32)
w = tf.reshape(w, [3,3])

sess = tf.Session()
sess.run(tf.global_variables_initializer())

def p(v):
    return sess.run(v)

def GetPruningIndex(src,std_cut):
    src_shape = tf.shape(src)
    src = tf.reshape(src,[-1])
    batch_mean, batch_var = tf.nn.moments(x = src,axes=[0])
    batch_std = tf.sqrt(batch_var)
    #print ('mean, var', p(batch_mean),p( batch_var),p( tf.sqrt(batch_var)))
    cut0 = batch_mean - batch_std * std_cut
    cut1 = batch_mean + batch_std * std_cut
    #print ('cut', p(cut0), p(cut1))
    prun_index0 = tf.cast(src<cut0, tf.float32)
    prun_index1 = tf.cast(src>cut1, tf.float32)
    prun_index =  tf.add(prun_index0, prun_index1)    
    return tf.reshape(prun_index,src_shape)

print (sess.run(w))
w_shape = w.get_shape().as_list()
std_cut = 0.5
pruning_mask = GetPruningIndex(w,std_cut)
print ('w_prun_index', p(pruning_mask))
print ('w_prun_index', p(tf.multiply(w,pruning_mask)))