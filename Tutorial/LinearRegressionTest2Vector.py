import tensorflow as tf
import numpy as np

# Predict Health Score from height and weight

x = [[1.50,0.40], 
     [1.60,0.55], 
     [1.70,0.65], 
     [1.80,0.65], 
     [1.65,0.99]] # [height Meter, weight Kilogram]
y = [0.3, 0.8, 0.95, 0.6, 0.05] #  Health Score 0.0~1.0

x = x / np.max(x)
y = y / np.max(y)

X = tf.placeholder(tf.float32, shape = [5,2])
Y = tf.placeholder(tf.float32, shape = [5])

w = tf.Variable(tf.random_uniform([2,1], -0.1,0.1))
bias = tf.Variable(0.1,[2,1])

def inference(inData):
    hypothesis = tf.matmul(inData, w) + bias
    return hypothesis

predict = inference(X)
cost = tf.reduce_mean( tf.square( Y - predict) )
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.1)
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
tf.summary.image('imgW', w,1);

for i in range(500):
         
    summary, _cost, _w, _bias, _train = sess.run(fetches = [merged, cost, w, bias,  train], feed_dict={X:x, Y:y})
    if i%30==0:
        print ('epoch %d, cost %f, bias %.3f' % (i, _cost, _bias))
        train_writer.add_summary(summary, i);

print ('test start')

x_test_data = [[1.55,0.90],
               [1.85,0.70], 
               [1.85,0.51],
               [1.65,0.62] 
               ] 

_predict = sess.run(inference(x_test_data)  )
print ('Score',_predict)
train_writer.close();

