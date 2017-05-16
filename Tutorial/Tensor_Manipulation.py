import tensorflow as tf
import numpy as np

def Print(input):
    print(sess.run(input))
t = tf.constant([1,2,3,4])
print('tf.shape(t)',tf.shape(t))
sess = tf.Session()
matrix1 = tf.constant([[1,2],[3,4]])
matrix2 = tf.constant([[2],[3]])

print('matrix1 shape',matrix1.shape)
print('matrix2 shape',matrix2.shape)

matrix_1x2 = tf.matmul(matrix1,matrix2)
print('matrix_1x2 shape',matrix_1x2.shape)

Print(matrix1)
Print(matrix2)
Print(matrix_1x2)
Print(tf.multiply(matrix1 ,matrix2))
Print(matrix1 * matrix2)
Print(matrix1 + matrix2) #Broadcasting

#One_hot

a = tf.constant([[1,0],[2,1]])
print('a')
Print(a)
b = tf.one_hot(a, depth=4)
print('b')
Print(b)
print('ones_like(b)')
Print(tf.ones_like(b))
print('zeros_like(b)')
Print(tf.zeros_like(b))

#boolean cast 
c = tf.constant([1,2,3,4,5])
c_big = c > 2
c_big_index = tf.cast(c_big, tf.int32) 
c_big_only = tf.multiply(c, c_big_index)
c_small_only = tf.multiply(c,1- c_big_index)
Print(c)
Print(c_big)
Print(c_big_index)
Print(c_big_only)
Print(c_small_only)

for x, y in zip([1,2],[3,4]):
    print (x,y)

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

min_x = np.mean(x,axis=1)
normal_x = x / np.mean(x,axis=0) 
print ('mean axis = 0',np.mean(x,axis=0) )
print ('mean axis = 1',np.mean(x,axis=1) )
print ('normal_x ',normal_x )