import tensorflow as tf

trimap = tf.constant([0,10,2,0,1,2.10])
trimap = tf.reshape(trimap, [1,1,2,3])
known = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)            

sess = tf.Session()
print (sess.run(trimap))
print (sess.run(tf.arg_max(trimap,3)))
print (sess.run(known))