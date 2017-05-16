import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

x = np.array([[0,0], [0,1], [1,0], [1,1]], dtype = np.float32)
y = np.array([[0],[1],[1],[0]], dtype = np.float32)  #XOR
#y = np.array([[0],[1],[1],[1]], dtype = np.float32) #OR
#y = np.array([[0],[1],[1],[0]], dtype = np.float32) #AND

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
w1 = tf.Variable(tf.random_normal([2,2]))
b1 = tf.Variable(tf.random_normal([2])) 
w2 = tf.Variable(tf.random_normal([2,1]))
b2 = tf.Variable(tf.random_normal([1])) 
               
def inference(input):
    g = tf.matmul(input , w1) + b1
    layer1 = tf.sigmoid(g)
    g2 = tf.matmul(layer1 , w2) +b2
    layer2 = tf.sigmoid(g2)
    return layer2

hypothesis = inference(X)
cost = tf.reduce_mean(tf.square(Y - hypothesis)) 
cost += 1e-5 * (tf.nn.l2_loss(w1) + tf.nn.l2_loss(w2))
train = tf.train.GradientDescentOptimizer(0.5).minimize(cost)
predicted = tf.cast(hypothesis > 0.5, dtype = tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted,Y), dtype= tf.float32))
sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()
for i in range(20000):
    hypo, l,acc, tr = sess.run(fetches = [hypothesis, cost,accuracy, train ], feed_dict = {X:x, Y:y})
    if i%100==0:
        print ('%d training loss : %f, acc : %f ' % (i, l,acc))
        plt.clf()
        plt.plot(x ,y, 'o')
        plt.plot(x ,hypo )
        plt.draw()
        plt.pause(0.001)    
        if acc > 0.99: 
            print ('accuracy > 0.99. OK End Train ',acc)
            break;

print ('w1')
print (sess.run( w1))
print ('w2')
print (sess.run( w2))
print ('predict')
hypo = sess.run( inference(x))
print (hypo )
print ('accuracy',acc)
plt.ioff()
plt.figure()
plt.plot(x,hypo,'o' )
plt.plot(x,hypo )
plt.show()