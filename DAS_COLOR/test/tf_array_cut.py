import tensorflow as tf

a = tf.constant(1.0, shape= [2,2,4])

b = a[:,:,0:1]
c = a[:,:,1:2]
print (a)
print (b)
print (c)