
import tensorflow as tf

def score(a,b):
    a = tf.argmax(a,1)
    b = tf.argmax(b,1)
    
    k = tf.cast(tf.equal(a,b),tf.int32)
    accuracy = tf.reduce_mean(k)
    eq = tf.equal(b * 2 - a,1)
    right = tf.reduce_sum(tf.cast(eq,tf.int64))
    precision = right / tf.reduce_sum(b)
    recall = right / tf.reduce_sum(a)
    f1 = 2 * precision*recall/(precision+recall)
    return f1


label = tf.constant([[1,0],[1,0],[0,1]])
predict = tf.constant([[1,0],[0,1],[0,1]])

a = tf.argmax(label,1)
b = tf.argmax(predict,1)
f1 = score(a,b)
sess= tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())
print ('f1', sess.run(f1))
print ('-----------------------')
print ('a',sess.run(a))
print ('b',sess.run(b))


k = tf.cast(tf.equal(a,b),tf.int32)
print ('k',sess.run(k))
accuracy = tf.reduce_mean(k)
print ('accuracy',sess.run(accuracy))
eq = tf.equal(b * 2 - a,1)
print ('eq',(eq))
right = tf.reduce_sum(tf.cast(eq,tf.int64))
precision = right / tf.reduce_sum(b)
recall = right / tf.reduce_sum(a)
f1 = 2 * precision*recall/(precision+recall)

print ('accuracy',sess.run( accuracy))
print ('precision', sess.run(precision))
print ('recall', sess.run(recall))
print ('f1', sess.run(f1))

iou = tf.metrics.mean_iou(label,predict,0)
print ('iou', sess.run(iou))
