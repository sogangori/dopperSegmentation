import tensorflow as tf
import numpy as np

# Predict Health Score from height and weight

x = [[2,1,1,0,0], 
     [1,3,1,0,0], 
     [1,3,4,3,1], 
     [0,0,1,2,1], 
     [0,0,1,2,2.0]] 
y = [0, 1, 2, 3, 4.0]

x /= np.max(x)
y /= np.max(y)

X = tf.placeholder(tf.float32, shape = [5,5])
Y = tf.placeholder(tf.float32, shape = [5])

w = tf.Variable(tf.random_uniform([5,1], -0.1,0.1))
w2 = tf.Variable(tf.random_uniform([5,1], -0.1,0.1))
w3 = tf.Variable(tf.random_uniform([5,1], -0.1,0.1))
bias = tf.Variable(0.1,[5,1])

def inference(inData):
    k = tf.matmul(tf.add(inData, w),w2)
    print ('k', k)
    hypothesis = k* w3 + bias
    return hypothesis

predict = inference(X)
cost = tf.reduce_mean( tf.square( Y - predict) )
cost += 1e-5* (tf.nn.l2_loss(w) + tf.nn.l2_loss(w2) + tf.nn.l2_loss(w3))
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = opt.minimize(cost)

init = tf.initialize_all_variables()
config=tf.ConfigProto()
config.log_device_placement= False
sess = tf.Session(config = config)
sess.run(init)

tf.summary.scalar('cost',cost)
tf.summary.scalar('w', w[0,0])
tf.summary.histogram('w', w)
tf.summary.histogram('bias', bias)
merged = tf.summary.merge_all();
train_writer = tf.summary.FileWriter('./train',sess.graph)

for i in range(500):
         
    summary, _cost, _w, _bias, _train = sess.run(fetches = [merged, cost, w, bias,  train], feed_dict={X:x, Y:y})
    if i%50==0:
        print ('epoch %d, cost %f, bias %.3f' % (i, _cost, _bias))
        train_writer.add_summary(summary, i);

print ('test start')

x_test_data = [[0,0,0,1,2.0],
               [0,0,1,2,1],                
               [0,3,4,3,0], 
               [1,3,1,0,0], 
               [2,1,0,0,0]
               ] 
x_test_data /= np.max(x_test_data)
_predict = sess.run(fetches = [predict], feed_dict={X:x_test_data} )
print ('Score',_predict)
train_writer.close();

