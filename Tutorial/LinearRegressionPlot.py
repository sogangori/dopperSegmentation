import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def inference(inData):
    hypothesis = inData * inData * w + b;
    return hypothesis

x_train = [1.50, 1.60, 1.70, 1.80, 1.65] # height Meter
y_train = [45, 55, 62, 70,65] #  weight Kilogram
x_test = [1.63, 1.88, 1.90, 1.65, 1.60, 1.55, 1.49] 

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

w = tf.Variable(1.0, name="w1")
b = tf.Variable(1.0, name="b1")

predict = inference(X)
cost = tf.square(Y - predict)
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.001)
train = opt.minimize(cost)

sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)

plt.ion()
for i in range(50):
    for j in range(len(x_train)):        
        _cost, _w, _b, _train = sess.run(fetches = [cost, w, b, train], feed_dict={X:x_train[j], Y:y_train[j]})
    predict_out= np.multiply(x_train , x_train) * sess.run(w) + sess.run(b)
    plt.clf()
    plt.plot(x_train ,y_train, 'o')
    plt.plot(x_train ,predict_out )
    plt.draw()
    plt.pause(0.001)    
        
    print ('epoch %d, cost %.2f, w %.2f' % (i, _cost, _w))

print ('train end. w = %f, b = %f, cost = %f' %(_w, _b, _cost))
print ('test start')

for j in range(4):      
    predict = inference(x_test[j])  
    _predict = sess.run(predict )
    print ('%d, height %.2f Meter, predict weight %.1f Kg' % (j, x_test[j], _predict))


predict_out= np.multiply(x_test, x_test) * sess.run(w) + sess.run(b)
plt.ioff()
plt.figure()
plt.plot(x_test,predict_out,'o' )
plt.plot(x_test,predict_out )
plt.show()