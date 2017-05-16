import tensorflow as tf

# Predict Health Score from height and weight

x = [[1.50,0.40], 
     [1.60,0.55], 
     [1.70,0.65], 
     [1.80,0.65], 
     [1.65,0.99]] # [height Meter, weight Kilogram]
y = [0.3, 0.8, 0.95, 0.6, 0.05] #  Health Score 0.0~1.0

X = tf.placeholder(tf.float32, shape = [2])
Y = tf.placeholder(tf.float32)

w1 = tf.Variable(0.1)
w2 = tf.Variable(0.1)
w3 = tf.Variable(0.1)
bias = tf.Variable(0.1)

def inference(inData):
    hypothesis = inData[0] * w1 + inData[1] * w2 + (inData[0] / inData[1]) * w3 + bias;
    return hypothesis

predict = inference(X)
cost = tf.square( Y - predict)
opt = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
train = opt.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for i in range(20):
    for j in range(len(x)):        
        _cost, _w1,_w2, _w3, _bias, _train = sess.run(fetches = [cost, w1, w2, w3, bias,  train], feed_dict={X:x[j], Y:y[j]})
        print ('step %d, epoch %d, cost %.5f, weights( %.3f,%.3f,%.3f,%.3f)' % (j, i, _cost, _w1, _w2,_w3, _bias))

print ('train end. w = %f' %(_w1))
print ('test start')

x_test_data = [[1.55,0.90],
               [1.65,0.62], 
               [1.85,0.70], 
               [1.85,0.51]] 

for j in range(len(x_test_data)):      
    predict = inference(x_test_data[j])  
    _predict = sess.run(predict)
    print ('%d, height %.2f, weight %.1f, Health Score %.2f' % (j, x_test_data[j][0], x_test_data[j][1], _predict))

