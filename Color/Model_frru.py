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
import Model_helper as helper

modelName = "./Color/weights/narrow_helper.pd"#narrow_
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

conv_s0_weights = tf.get_variable("s0", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_s0_biases = tf.Variable(tf.zeros([depth0]))

conv_t0_weights = tf.get_variable("t0", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_t0_biases = tf.Variable(tf.zeros([depth0]))

conv_p0_weights = tf.get_variable("p0", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_p0_biases = tf.Variable(tf.zeros([depth0]))

conv_x0_weights = tf.get_variable("x0", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_x0_biases = tf.Variable(tf.zeros([depth0]))

conv_xx0_weights = tf.get_variable("xx0", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_xx0_biases = tf.Variable(tf.zeros([depth0]))

conv_x2_weights = tf.get_variable("x2", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_x2_biases = tf.Variable(tf.zeros([depth0]))

conv_p2_weights = tf.get_variable("p2", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_p2_biases = tf.Variable(tf.zeros([depth0]))

conv_t2_weights = tf.get_variable("t2", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_t2_biases = tf.Variable(tf.zeros([depth0]))

conv_s2_weights = tf.get_variable("s2", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_s2_biases = tf.Variable(tf.zeros([depth0]))

conv_m2_weights = tf.get_variable("m2", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_m2_biases = tf.Variable(tf.zeros([depth0]))

conv_l2_weights = tf.get_variable("l2", shape=[3, 3, depth0+depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_l2_biases = tf.Variable(tf.zeros([depth0]))

conv_l3_weights = tf.get_variable("l3", shape=[3, 3, depth0, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
conv_l3_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))


def inference(inData, train=False):
    helper.isDrop = train
    helper.keep_prop = 0.6
    
    featureMap = []
    inData = tf.multiply(inData ,1.0)
    if train: inData = helper.Gaussian_noise_Add(inData, 0.1,0.3)
    
    #1/2
    in2 = tf.nn.avg_pool(inData,pool_stride2,strides=pool_stride2,padding='SAME')     
    feature1 = pool = helper.conv2dRelu(in2,conv_l0_weights,conv_l0_biases)
    
    #1/4
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2d(pool,conv_m0_weights,conv_m0_biases)   
    feature2 = tf.nn.relu(pool)
    feature1_shape = up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2]) 
    feature1 = tf.nn.relu(tf.add(feature1 , pool))

    #1/8
    feature1down = tf.nn.max_pool(feature1,[1, 4, 4, 1],strides=[1, 4, 4, 1],padding='SAME')    
    feature2down = tf.nn.max_pool(feature2,pool_stride2,strides=pool_stride2,padding='SAME')   
    pool = tf.concat([feature2down, feature1down],3)
    pool = helper.conv2d(pool,conv_s0_weights,conv_s0_biases)   
    feature3 = tf.nn.relu(pool)
    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2]) 
    feature1 = tf.nn.relu(tf.add(feature1 , pool))

    #1/16
    feature1down = tf.nn.max_pool(feature1,[1, 8, 8, 1],strides=[1, 8, 8, 1],padding='SAME')    
    feature3down = tf.nn.max_pool(feature3,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = tf.concat([feature3down, feature1down],3)
    pool = helper.conv2d(pool,conv_t0_weights,conv_t0_biases)   
    feature4 = tf.nn.relu(pool)
    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2]) 
    feature1 = tf.nn.relu(tf.add(feature1 , pool))

    feature1down = tf.nn.max_pool(feature1,[1, 16, 16, 1],strides=[1, 16, 16, 1],padding='SAME')    
    feature4down = tf.nn.max_pool(feature4,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = tf.concat([feature4down, feature1down],3)
    pool = helper.conv2d(pool,conv_p0_weights,conv_p0_biases)   
    feature5 = tf.nn.relu(pool)
    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2]) 
    feature1 = tf.nn.relu(tf.add(feature1 , pool))
     
    feature1down = tf.nn.max_pool(feature1,[1, 32, 32, 1],strides=[1, 32, 32, 1],padding='SAME')    
    feature5down = tf.nn.max_pool(feature5,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = tf.concat([feature5down, feature1down],3)
    pool = helper.conv2d(pool, conv_p0_weights,conv_p0_biases)   
    feature6 = tf.nn.relu(pool)
    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2]) 
    feature1 = tf.nn.relu(tf.add(feature1 , pool))

    featureMap.append(inData)   
    featureMap.append(feature1)
    featureMap.append(feature2)
    featureMap.append(feature3)
    featureMap.append(feature4)
    featureMap.append(feature5)
    featureMap.append(feature6)
    featureMap.append(pool)
        
    up_shape = feature5.get_shape().as_list()
    feature_up = helper.resize(feature6, up_shape[1],up_shape[2])    
    feature1down = tf.nn.max_pool(feature1,[1, 16, 16, 1],strides=[1, 16, 16, 1],padding='SAME')    
    pool = tf.nn.relu(tf.concat([feature1down, feature_up],3)) 
    pool = helper.conv2d(tf.nn.relu(pool),conv_p2_weights,conv_p2_biases)   
    feature5 = tf.nn.relu(pool)
    poolUp = helper.resize(pool, feature1_shape[1],feature1_shape[2])    
    feature1 = tf.nn.relu(tf.add(feature1 ,poolUp))
    featureMap.append(pool)

    up_shape = feature4.get_shape().as_list()
    pool = helper.resize(feature5, up_shape[1],up_shape[2])    
    feature1down = tf.nn.max_pool(feature1,[1, 8, 8, 1],strides=[1, 8, 8, 1],padding='SAME')    
    pool = tf.nn.relu(tf.concat([feature1down, pool],3)) 
    pool = helper.conv2d(pool,conv_t2_weights,conv_t2_biases)   
    poolUp = helper.resize(pool, feature1_shape[1],feature1_shape[2])    
    feature1 = tf.nn.relu(tf.add(feature1 ,poolUp))
    featureMap.append(pool)

    up_shape = feature3.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    feature1down = tf.nn.max_pool(feature1,[1, 4, 4, 1],strides=[1, 4, 4, 1],padding='SAME')    
    pool = tf.nn.relu(tf.concat([feature1down, pool],3)) 
    pool = helper.conv2d(pool,conv_s2_weights,conv_s2_biases)   
    feature4 = tf.nn.relu(pool)
    poolUp = helper.resize(pool, feature1_shape[1],feature1_shape[2])    
    feature1 = tf.nn.relu(tf.add(feature1 ,poolUp))
    featureMap.append(pool)

    up_shape = feature2.get_shape().as_list()
    pool = helper.resize(feature4, up_shape[1],up_shape[2])    
    feature1down = tf.nn.max_pool(feature1,[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')    
    pool = tf.nn.relu(tf.concat([feature1down, pool],3)) 
    pool = helper.conv2d(pool,conv_m2_weights,conv_m2_biases)   
    feature3 = tf.nn.relu(pool)
    poolUp = helper.resize(pool, feature1_shape[1],feature1_shape[2])    
    feature1 = tf.nn.relu(tf.add(feature1 ,poolUp))  
    featureMap.append(pool)
        
    up_shape = feature1.get_shape().as_list()
    feature1down = tf.nn.max_pool(poolUp,[1, 2, 2, 1],strides=[1, 2, 2, 1],padding='SAME')    
    pool = tf.nn.relu(tf.concat([feature1down, pool],3)) 
    pool = helper.conv2d(pool,conv_l2_weights,conv_l2_biases)   
    feature3 = tf.nn.relu(pool)
    poolUp = helper.resize(pool, feature1_shape[1],feature1_shape[2])    
    feature1 = tf.nn.relu(tf.add(feature1 ,poolUp))  
        
    pool = helper.conv2dRelu(pool,conv_l3_weights,conv_l3_biases)
  
    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
    if not train: 
        pool = tf.reshape(pool, [-1,LABEL_SIZE_C])
        pool = tf.nn.softmax(pool)
        input_shape[3] = LABEL_SIZE_C
        pool = tf.reshape(pool, input_shape)
    return pool,featureMap; 