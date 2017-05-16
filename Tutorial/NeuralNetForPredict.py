import numpy as np
import tensorflow as tf

#feature [height, weight, foot size]
#label [0 = woman, 1 = man]
#Goal : Predict man or woman for x_test

x_test = np.array([[162,50,240],#woman
              [156,49,225],
              [163,53,255],
              [167,65,260],
              [175,68,270],            
              [180,70,275] 
              ], dtype = np.float32)

y_test = np.array([[0],
              [0],
              [0],
              [1],
              [1],
              [1]
              ], dtype = np.float32)  

x = np.array([[160,50,250],#woman
              [156,49,225],
              [155,48,220],
              [158,49,230], 
              [167,65,260], 
              [167,55,250], 
              [171,55,240],
              [178,60,250],
              [181,55,240],
              [158,55,240],#man 
              [160,55,250], 
              [165,50,250], 
              [170,60,260], 
              [175,65,265],
              [180,70,275],
              [185,85,280],
              ], dtype = np.float32)

y = np.array([[0],#woman
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [0],
              [1],#man 
              [1],
              [1],
              [1],
              [1],
              [1],
              [1]
              ], dtype = np.float32)  

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

# Modify Start  ---------------
x = x / np.mean(x,axis=0)
x_test = x_test / np.mean(x_test,axis=0)
learning_rate = 5
feature = 16
feature2 = 8
with tf.name_scope("layer1") as scope:
    w1 = tf.Variable(tf.random_normal([3,feature]))
    b1 = tf.Variable(tf.random_normal([feature])) 
with tf.name_scope("layer2") as scope:
    w2 = tf.Variable(tf.random_normal([feature,1]))
    b2 = tf.Variable(tf.random_normal([1])) 

def inference(input):
    g = tf.matmul(input , w1) + b1
    layer1 = tf.sigmoid(g)
    g2 = tf.matmul(layer1 , w2) +b2
    layer2 = tf.sigmoid(g2)
    return layer2

# Modify End  ---------------


hypothesis = inference(X)
cost = tf.reduce_mean(tf.square(Y - hypothesis)) 
cost += 1e-5 * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
train = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype= tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())
tf.summary.histogram("w1",w1)
tf.summary.histogram("w2",w2)
tf.summary.histogram("b1",b1)
tf.summary.histogram("b2",b2)
tf.summary.scalar("cost",cost)
tf.summary.scalar("accuracy",accuracy)
summary = tf.summary.merge_all();
writer = tf.summary.FileWriter('./log') #tensorboard --logdir=./log
writer.add_graph(sess.graph)  
for i in range(10000):
    s, hypo, l,acc, tr = sess.run(fetches = [summary , hypothesis, cost,accuracy, train ], feed_dict = {X:x, Y:y})
    if i%100==0:
        writer.add_summary(s, i/100);
        print ('%d training loss : %f, acc : %f ' % (i, l,acc))
        if acc > 0.99: 
            print ('accuracy > 0.99. OK End Train ',acc)
            break;

print ('w1')
print (sess.run( w1))
print ('w2')
print (sess.run( w2))
print ('predict')
print ('Train accuracy : %f, cost : %f' %(acc, l))
hypo, l,acc = sess.run(fetches=[hypothesis, cost , accuracy], feed_dict={X:x_test, Y:y_test})
print (hypo )
print ('Test  accuracy : %f, cost : %f' %(acc, l))
