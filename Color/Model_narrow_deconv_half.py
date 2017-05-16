from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import time
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

modelName = "./Color/weights/half2.pd"
logName ="./Color/weights/logs_half2"

#$ tensorboard --logdir=/home/digits/workspace_git/cmode_tensorflow/train0/weights/logs_color1
#browser  http://0.0.0.0:6006

isDropout = True
LABEL_SIZE_C = 2
NUM_CHANNELS_In= 3

keep_prop = 0.5

pool_stride2 =[1, 2, 2, 1]

#depth 2, Aug x3~x4 : 83%, 80% loss 0.041x 
#depth 2, deconv : 85%, 82% loss 0.118x
depth0 = 2

conv_l0_weights  = tf.get_variable("w1", shape=[3, 3, NUM_CHANNELS_In, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_l0_biases = tf.Variable(tf.zeros([depth0]))

conv_m0_weights = tf.get_variable("m0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_m0_biases = tf.Variable(tf.zeros([depth0]))

conv_s0_weights = tf.get_variable("s0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_s0_biases = tf.Variable(tf.zeros([depth0]))

conv_t0_weights = tf.get_variable("t0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_t0_biases = tf.Variable(tf.zeros([depth0]))

conv_p0_weights = tf.get_variable("p0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_p0_biases = tf.Variable(tf.zeros([depth0]))

conv_x0_weights = tf.get_variable("x0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_x0_biases = tf.Variable(tf.zeros([depth0]))

conv_xx0_weights = tf.get_variable("xx0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_xx0_biases = tf.Variable(tf.zeros([depth0]))


conv_l2_weights = tf.get_variable("l2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_l2_biases = tf.Variable(tf.zeros([depth0]))

conv_l3_weights = tf.get_variable("l3", shape=[3, 3, depth0, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
conv_l3_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))

dconv_0_weights = tf.get_variable("d0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_0_biases = tf.Variable(tf.zeros([depth0]))
dconv_1_weights = tf.get_variable("d1", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_1_biases = tf.Variable(tf.zeros([depth0]))
dconv_2_weights = tf.get_variable("d2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_2_biases = tf.Variable(tf.zeros([depth0]))
dconv_3_weights = tf.get_variable("d3", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_3_biases = tf.Variable(tf.zeros([depth0]))
dconv_4_weights = tf.get_variable("d4", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_4_biases = tf.Variable(tf.zeros([depth0]))
dconv_5_weights = tf.get_variable("d5", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_5_biases = tf.Variable(tf.zeros([depth0]))

def resize(src, dstH, dstW,interpol = 1):
    
    if interpol == 0: return tf.image.resize_nearest_neighbor(src, [dstH, dstW])
    elif interpol == 1: return tf.image.resize_bilinear(src, [dstH, dstW])
    else: return tf.image.resize_bicubic(src, [dstH, dstW])

def conv2d(src, weights, bias):
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    return tf.nn.bias_add(conv,bias)

def Gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape = input_layer.get_shape(), mean = 0.0, stddev = std, dtype = tf.float32) 
    return input_layer + noise

def conv2dRelu(src, weights, bias, isDrop= False):
    if isDrop: src = tf.nn.dropout(src, keep_prop, seed=time.time())
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    conv = tf.nn.bias_add(conv,bias)    
    return tf.nn.relu(conv) 

def upConv(src, weights, bias,up_shape):
    output = tf.nn.conv2d_transpose(src,weights,output_shape=up_shape, strides=[1, 2, 2, 1],padding='SAME')
    output = tf.nn.bias_add(output,bias)  
    return output

def inference(inData, train=False):
    isDrop = train and isDropout
    
    #1/2
    inData = tf.multiply(inData ,1.0)
    if train: inData =  Gaussian_noise_layer(inData, 0.2)
    in2 = tf.nn.avg_pool(inData,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature1 = pool = conv2dRelu(in2,conv_l0_weights,conv_l0_biases,isDrop)
    
    #1/4
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature2 = pool = conv2dRelu(pool,conv_m0_weights,conv_m0_biases)   

    #1/8
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = conv2dRelu(pool,conv_s0_weights,conv_s0_biases)   

    #1/16
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature4 = pool = conv2d(pool,conv_t0_weights,conv_t0_biases)   

    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature5 = pool = conv2d(pool,conv_p0_weights,conv_p0_biases)   
    
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature6 = pool = conv2d(pool,conv_x0_weights,conv_x0_biases)   

    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = conv2d(pool,conv_xx0_weights,conv_xx0_biases)   
    
    pool = upConv(pool, dconv_5_weights,dconv_5_biases, feature6.get_shape().as_list() )
    pool = tf.add(feature6, pool) 
    pool = tf.nn.relu(pool)   

    pool = upConv(pool, dconv_4_weights,dconv_4_biases, feature5.get_shape().as_list())
    pool = tf.add(feature5, pool) 
    pool = tf.nn.relu(pool)   
 
    pool = upConv(pool, dconv_3_weights,dconv_3_biases, feature4.get_shape().as_list())
    pool = tf.add(feature4, pool) 
    pool = tf.nn.relu(pool)   

    pool = upConv(pool, dconv_2_weights,dconv_2_biases, feature3.get_shape().as_list())
    pool = tf.add(feature3, pool) 
    pool = tf.nn.relu(pool)   

    pool = upConv(pool, dconv_1_weights,dconv_1_biases, feature2.get_shape().as_list())
    pool = tf.add(feature2, pool) 
    pool = tf.nn.relu(pool)    

    pool = upConv(pool, dconv_0_weights,dconv_0_biases, feature1.get_shape().as_list())
    pool = tf.add(feature1, pool) 
    pool = conv2dRelu(pool,conv_l2_weights,conv_l2_biases,isDrop)     
    pool = conv2dRelu(pool,conv_l3_weights,conv_l3_biases,isDrop)
  
    input_shape = inData.get_shape().as_list()
    pool = resize(pool,input_shape[1] ,input_shape[2])
    reshape = tf.reshape(pool, [-1,LABEL_SIZE_C])
    if not train: reshape = tf.nn.softmax(reshape)
    return reshape; 

def regullarizer():
    
    loss = tf.nn.l2_loss(conv_l0_weights) + tf.nn.l2_loss(conv_l2_weights) + tf.nn.l2_loss(conv_l3_weights)
    loss +=tf.nn.l2_loss(conv_m0_weights) 
    loss +=tf.nn.l2_loss(conv_s0_weights) 
    loss +=tf.nn.l2_loss(conv_t0_weights) 
    loss +=tf.nn.l2_loss(conv_p0_weights) 
    loss +=tf.nn.l2_loss(conv_x0_weights) 
    loss +=tf.nn.l2_loss(conv_xx0_weights)
    loss +=tf.nn.l2_loss(dconv_0_weights) + tf.nn.l2_loss(dconv_1_weights)
    loss +=tf.nn.l2_loss(dconv_2_weights) + tf.nn.l2_loss(dconv_3_weights)
    loss +=tf.nn.l2_loss(dconv_4_weights) + tf.nn.l2_loss(dconv_5_weights)
    return loss
