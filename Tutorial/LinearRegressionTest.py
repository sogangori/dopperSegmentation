import tensorflow as tf
import numpy as np

mode_b_path ="./modelB"
mode_w_path ="./modelW"
is_load_0 = not True
is_load_1 = not True
is_train_0 =  True
is_train_1 =  True

def inference_network_0(inData):
    hypothesis = inData * inData * w + b;
    return hypothesis

def inference_network_1(inData):
    hypothesis = inData * w2 + b2;
    return hypothesis

x_train = [1.50, 1.60, 1.70, 1.80, 1.65] # height Meter
y_train = [45, 55, 62, 70,65] #  weight Kilogram
x_test = [1.55, 1.65, 1.75, 1.85] 

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

with tf.variable_scope('model1'):
    w = tf.Variable(1.0)
    b = tf.Variable(0.0)

with tf.variable_scope('model2'):
    w2 = tf.Variable(1.0)
    b2 = tf.Variable(0.0)

predict_0 = inference_network_0(X)
predict_1 = inference_network_1(predict_0)

regularize_term_0 = tf.nn.l2_loss(w)
regularize_term_1 = tf.nn.l2_loss(w2)

cost_0 = tf.square(Y - predict_0)
cost_1 = tf.square(Y - predict_1)

cost_0 += 1e-4 * regularize_term_0 
cost_1 += 1e-4 * regularize_term_1

opt_0 = tf.train.GradientDescentOptimizer(learning_rate = 0.01)
opt_1 = tf.train.GradientDescentOptimizer(learning_rate = 0.0001)

variable_model_1 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'model1')
variable_model_2 = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'model2')
train_0 = opt_0.minimize(cost_0, var_list=variable_model_1)
train_1 = opt_1.minimize(cost_1, var_list=variable_model_2)

sess = tf.Session()
saver_w = tf.train.Saver(variable_model_1)
saver_b = tf.train.Saver(variable_model_2)

init = tf.global_variables_initializer()
sess.run(init)

if is_load_0:
    saver_w.restore(sess, mode_w_path)
    print ('Model Restored w = %f' % (sess.run(w)))

if is_load_1:
    saver_b.restore(sess, mode_b_path)
    print ('Model Restored b = %f' % (sess.run(b)))

for i in range(10):
    for j in range(len(x_train)):
        feed_dict_data = {X:x_train[j], Y:y_train[j]}
        if is_train_0: sess.run(fetches = [train_0], feed_dict = feed_dict_data)
        if is_train_1: sess.run(fetches = [train_1], feed_dict = feed_dict_data)
    
    _cost, _cost_1, _w, _b, _w2, _b2 = sess.run(fetches = [cost_0, cost_1, w, b, w2, b2], feed_dict={X:x_train[j], Y:y_train[j]})
    predict_out= np.multiply(x_train , x_train) * sess.run(w) + sess.run(b)
   
    print ('epoch %d, cost %.2f, cost_1 %.2f, w %.4f, b %.4f, w2 %.4f, b2 %.4f' % (i, _cost, _cost_1, _w, _b, _w2, _b2))

print ('train end. cost = %f, cost_1 = %f,w = %f, b = %f' %(_cost, _cost_1,_w, _b))
print ('test start')
save_w_path = saver_w.save(sess, mode_w_path)
save_b_path = saver_b.save(sess, mode_b_path)
print ('Model saved in File : %s' % save_w_path,save_b_path  )

for j in range(len(x_test)):      
    predict = inference_network_1(inference_network_0(x_test[j]))
    _predict = sess.run(predict )
    print ('%d, height %.2f Meter, predict weight %.1f Kg' % (j, x_test[j], _predict))

