﻿from __future__ import absolute_import
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

modelName = "./Color/weights/roi.pd" 
LABEL_SIZE_C = 2
NUM_CHANNELS_In= 3
pool_stride2 =[1, 2, 2, 1]
depth0 = 3

conv_roi_4_weights = tf.get_variable("roi4", shape=[3, 3, depth0, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
conv_roi_4_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))

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

conv_l3_weights = tf.get_variable("l3", shape=[3, 3, depth0+1, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
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
    feature2 = pool = helper.conv2dRelu(pool,conv_m0_weights,conv_m0_biases)   

    #1/8
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = helper.conv2dRelu(pool,conv_s0_weights,conv_s0_biases)   

    roi = helper.conv2dRelu(pool,conv_roi_4_weights,conv_roi_4_biases)        

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
    pool = helper.conv2d(pool,conv_s2_weights,conv_s2_biases)   
    featureMap.append(pool)

    up_shape = feature2.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature2, pool)) 
    pool = helper.conv2d(pool,conv_m2_weights,conv_m2_biases)   
    featureMap.append(pool)
        
    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature1, pool)) 
    pool = helper.conv2dRelu(pool,conv_l2_weights,conv_l2_biases)     

    up_shape = in2.get_shape().as_list()
    roi = helper.resize(roi, up_shape[1],up_shape[2])    
    roi_re = tf.reshape(roi, [-1,LABEL_SIZE_C])
    roi_re = tf.nn.softmax(roi_re)
    roi_re = tf.reshape(roi_re, roi.get_shape().as_list())
    roi_0 = tf.reshape(roi_re[:,:,:,0], [up_shape[0],up_shape[1],up_shape[2],1])
    pool = tf.concat([pool, roi_0],3)

    pool = helper.conv2dRelu(pool,conv_l3_weights,conv_l3_biases)
  
    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
    roi = helper.resize(roi,input_shape[1] ,input_shape[2])
    if not train: 
        pool = tf.reshape(pool, [-1,LABEL_SIZE_C])
        pool = tf.nn.softmax(pool)
        input_shape[3] = LABEL_SIZE_C
        pool = tf.reshape(pool, input_shape)
    return pool,roi,featureMap; 