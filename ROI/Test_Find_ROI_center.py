import tensorflow as tf
import numpy as np

def GetNonZeroYH(data):
    c = np.shape(data)[0]
    outdata = np.zeros(shape=(c,2), dtype=np.float32)
    for i in range(c):
        src = data[i]
        print ('src', np.shape(src))
        src_col_sum = np.sum(src, axis=1)
        print ('src_col_sum', np.shape(src_col_sum))
        print (src_col_sum)
    
        h = len(src_col_sum)
        y0 = 0
        y1 = h
        for y in range(h):
            if src_col_sum[y]>0:
                y0 = y
                break
        for y in range(y1):
            idx = y1-y-1
            if src_col_sum[idx]>0:
                y1 = idx
                break

        print ('height: %d, y0~y1 = %d ~ %d' %(h, y0,y1))
        roiY = (y0+y1)/2
        roiH = y1-y0+1
        normal_roiY = 1.0* roiY / h
        normal_roiH = 1.0* roiH / h
        print ('roiY:%d, roiH:%d,  normal : %g, %g' %(roiY,roiH,normal_roiY,normal_roiH))
        outdata[i] = [normal_roiY,normal_roiH]
    return outdata

x = [[[0,0,0,0,0], 
     [0,1,1,0,0], 
     [1,1,1,1,1], 
     [0,0,1,1,1], 
     [0,0,0,0,0]] ,
    [[0,1,0,0,0], 
     [0,1,1,0,0], 
     [1,1,1,1,1], 
     [0,0,0,0,0], 
     [0,0,0,0,0]] ]

y = GetNonZeroYH(x)

X = tf.placeholder(tf.float32, shape=[None, 5,5])
Y = tf.placeholder(tf.float32, shape=[None, 2])

w = tf.Variable(tf.constant(0.1, shape=[25,2]))
b = tf.Variable(tf.constant(0.0, shape=[2]))

x_re = tf.reshape(X,shape=[-1,25])
h =  tf.nn.sigmoid( tf.matmul(x_re,w) + b )
cost = tf.reduce_mean( tf.square(Y - h) )
train = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
epoch = 800

sess = tf.Session()
sess.run(tf.global_variables_initializer())
print (y)
for i in range(epoch):
    _, l, hyper = sess.run([train, cost,h], feed_dict = {X:x, Y:y})

    print('%d, l:%g' %(i,l) )
    if i%(epoch/10)==0:
        print(hyper)

print ('y', y)
print('h',hyper)


