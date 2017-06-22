import tensorflow as tf
import numpy as np

x = np.array([[[1,10],[2,20]],[[3,30],[4,40]]])
y = np.array([[21],[42]])
X = tf.placeholder(tf.float32, [None,2,2])
Y = tf.placeholder(tf.float32, [None,1])
depth = 2
beta= tf.Variable(tf.constant(0.0, shape=[depth]))
gamma= tf.Variable(tf.constant(1.0, shape=[depth]))

batch_mean, batch_var = tf.nn.moments(x=X,axes=[0])
#batch_mean = tf.constant(1.0)
#batch_var = tf.constant(4.0)
out = tf.nn.batch_normalization(x=X, mean=batch_mean, variance=batch_var,offset=beta,scale=gamma,variance_epsilon=1e-3)    
cost = tf.reduce_mean(tf.square(Y-out))
train = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize(cost)
sess=tf.Session()
sess.run(tf.global_variables_initializer())
count = 4
for i in range(3):

    #batch = np.reshape(x[i%count,:,:],[-1,2,2])
    #batchY =  np.reshape(y[i%count,:,:],[-1,2,2])
    batch = x
    batchY = y
    _,out_,cost_, b,g,bm,bv,o= sess.run([train,out,cost, beta, gamma, batch_mean, batch_var, out],feed_dict = {X:batch,Y:batchY })
    print ('%d, cost:%g' % (i,cost_))
    print ('in' ,batch)
    print ('out' ,out_)
    print ('beta',b)
    print ('gamma',g)
    print ('batch_mean',bm)
    print ('batch_var',bv)
    print ('out',o)
    print ('')

b,g= sess.run([beta, gamma],feed_dict = {X:batch,Y:batchY })
print ('final beta',b)
print ('final gamma',g)
saver = tf.train.Saver()
saver.save(sess,'./bn_weight')