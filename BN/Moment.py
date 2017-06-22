import numpy as np
import tensorflow as tf

x = np.array([[1,10],[2,20],[3,30],[5,50]])
X = tf.placeholder(tf.float32, [None,2])
batch_mean, batch_var = tf.nn.moments(x=X,axes=[0])
print (x.shape)
print (x)

sess=tf.Session()
sess.run(tf.global_variables_initializer())
x0 = x[0:2]
x1 = x[2:]
bm,bv = sess.run([batch_mean, batch_var],feed_dict = {X:x0})
print ('batch_mean',bm)
print ('batch_var', bv)

bm,bv = sess.run([batch_mean, batch_var],feed_dict = {X:x1})
print ('batch_mean',bm)
print ('batch_var', bv)

#print ('mean axis=0 ',np.mean(x,axis=0))
#print ('mean axis=1 ',np.mean(x,axis=1))
