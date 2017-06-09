from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
#http://davidstutz.de/batch-normalization-in-tensorflow/
import gzip
import os
import sys
import time
import numpy
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
import Model_helper as helper

modelName = "./Color/weights/bn_iou_narrow.pd"
LABEL_SIZE_C = 2
NUM_CHANNELS_In= 3
pool_stride2 =[1, 2, 2, 1]
depth0 = 3

#depth 1 : 86%, 82% loss 0.15x shape bad
#depth 2, Aug x2 : 85%, 82% loss 0.114x shape good
#depth 2, Aug x3~x4 : 83%, 80% loss 0.041x 

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

conv_x2_weights = tf.get_variable("x2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_x2_biases = tf.Variable(tf.zeros([depth0]))

conv_p2_weights = tf.get_variable("p2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_p2_biases = tf.Variable(tf.zeros([depth0]))

conv_t2_weights = tf.get_variable("t2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_t2_biases = tf.Variable(tf.zeros([depth0]))

conv_s2_weights = tf.get_variable("s2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_s2_biases = tf.Variable(tf.zeros([depth0]))

conv_m2_weights = tf.get_variable("m2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_m2_biases = tf.Variable(tf.zeros([depth0]))

conv_l2_weights = tf.get_variable("l2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_l2_biases = tf.Variable(tf.zeros([depth0]))

conv_l3_weights = tf.get_variable("l3", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
conv_l3_biases = tf.Variable(tf.zeros([1]))

beta = tf.Variable(tf.constant(0.0, shape=[depth0]),name='beta', trainable=True)
gamma = tf.Variable(tf.constant(1.0, shape=[depth0]),name='gamma', trainable=True)

beta1 = tf.Variable(tf.constant(0.0, shape=[depth0]),name='beta1', trainable=True)
gamma1 = tf.Variable(tf.constant(1.0, shape=[depth0]),name='gamma1', trainable=True)

beta2 = tf.Variable(tf.constant(0.0, shape=[depth0]),name='beta2', trainable=True)
gamma2 = tf.Variable(tf.constant(1.0, shape=[depth0]),name='gamma2', trainable=True)

beta3 = tf.Variable(tf.constant(0.0, shape=[depth0]),name='beta3', trainable=True)
gamma3 = tf.Variable(tf.constant(1.0, shape=[depth0]),name='gamma3', trainable=True) 

step = 0
def inference(inData, train = False):
    helper.isDrop = False
    helper.keep_prop = 0.6
    
    featureMap = []
    in2 = inData = tf.multiply(inData ,1.0)
    if step%3==1:  in2= tf.nn.avg_pool(inData,pool_stride2,strides=pool_stride2,padding='SAME')
    #elif step%3==2:in2= tf.nn.avg_pool(inData,pool_stride3,strides=pool_stride3,padding='SAME')
    feature1 = pool = helper.conv2dBN(in2,conv_l0_weights,beta,gamma,train)

    #1/4
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature2 = pool = helper.conv2dBN(pool,conv_m0_weights,beta1,gamma1, train)   

    #1/8
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = helper.conv2dBN(pool,conv_s0_weights,beta2,gamma2, train)   

    #1/16
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature4 = pool = helper.conv2dRelu(pool,conv_t0_weights,conv_t0_biases)   

    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature5 = pool = helper.conv2dRelu(pool,conv_p0_weights,conv_p0_biases)   
    
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature6 = pool = helper.conv2dRelu(pool,conv_x0_weights,conv_x0_biases)   

    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2d(pool,conv_xx0_weights,conv_xx0_biases)   
        
    featureMap.append(inData)   
    featureMap.append(feature1)
    featureMap.append(feature2)
    featureMap.append(feature3)
    featureMap.append(feature4)
    featureMap.append(feature5)
    featureMap.append(feature6)
    featureMap.append(pool)

    up_shape = feature6.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature6, pool))
    pool = helper.conv2d(pool,conv_x2_weights,conv_x2_biases)   
    featureMap.append(pool)

    up_shape = feature5.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature5, pool)) 
    pool = helper.conv2d(pool,conv_p2_weights,conv_p2_biases)   
    featureMap.append(pool)

    up_shape = feature4.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature4, pool)) 
    pool = helper.conv2d(pool,conv_t2_weights,conv_t2_biases)   
    featureMap.append(pool)

    up_shape = feature3.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature3, pool)) 
    pool = helper.conv2dBN(pool,conv_m2_weights,beta3,gamma3, train)
    pool = helper.conv2dRelu(pool,conv_l2_weights,conv_l2_biases)     
    pool = helper.conv2d(pool,conv_l3_weights,conv_l3_biases)
    pool = tf.nn.sigmoid(pool)
  
    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
  
    return pool,featureMap; 