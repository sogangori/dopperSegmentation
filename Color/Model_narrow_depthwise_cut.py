﻿from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import numpy
from six.moves import urllib
from six.moves import xrange 
import tensorflow as tf
import Model_helper as helper

modelName = "./Color/weights/depthwise3cut.pd"

isDropout = True
LABEL_SIZE_C = 2
NUM_CHANNELS_In= 3

keep_prop = 0.6
pool_stride2 =[1, 2, 2, 1]

#depth 2, Aug x3~x4 : 83%, 80% loss 0.041x 
#depth 3, deconv : 85%, 82% loss 0.118x
#depthwise 3, deconv : 86%, 83% loss 0.0118x
#depthwise 3, deconv_cut1 : 86%, 83% entropy 0.119x
#depthwise 3, deconv_cut2 : 85%, 82% entropy 0.132x not good enough
depth0 = 3

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

conv_p2_weights = tf.get_variable("p2", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
conv_p2_biases = tf.Variable(tf.zeros([depth0]))

conv_t2_weights = tf.get_variable("t2", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
conv_t2_biases = tf.Variable(tf.zeros([depth0]))

conv_s2_weights = tf.get_variable("s2", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
conv_s2_biases = tf.Variable(tf.zeros([depth0]))

conv_m2_weights = tf.get_variable("m2", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
conv_m2_biases = tf.Variable(tf.zeros([depth0]))

conv_l2_weights = tf.get_variable("l2", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
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

def inference(inData, train=False):
    isDrop = train and isDropout
    featureMap = []
    #1/2
    inData = tf.multiply(inData ,1.0)
    if train: inData =  helper.Gaussian_noise_layer(inData, 0.3)
    in2 = tf.nn.avg_pool(inData,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature1 = pool = helper.conv2dRelu(in2,conv_l0_weights,conv_l0_biases,isDrop,keep_prop)
    
    #1/4
    some,pos0 = tf.nn.max_pool_with_argmax(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature2 = pool = helper.conv2dRelu(pool,conv_m0_weights,conv_m0_biases,isDrop,keep_prop)

    #1/8
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = helper.conv2dRelu(pool,conv_s0_weights,conv_s0_biases,isDrop,keep_prop)   

    #1/16
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature4 = pool = helper.conv2d(pool,conv_t0_weights,conv_t0_biases)   

    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2d(pool,conv_p0_weights,conv_p0_biases)   
    
    featureMap.append(inData)   
    featureMap.append(feature1)
    featureMap.append(feature2)
    featureMap.append(feature3)
    featureMap.append(feature4)
    featureMap.append(pool)
 
    pool = helper.upConv(pool, dconv_3_weights,dconv_3_biases, feature4.get_shape().as_list())
    pool = tf.nn.relu(tf.add(feature4, pool) )
    pool = helper.depthwiseConv2dRelu(pool,conv_t2_weights,conv_t2_biases)   
    featureMap.append(pool)

    pool = helper.upConv(pool, dconv_2_weights,dconv_2_biases, feature3.get_shape().as_list())
    pool = tf.nn.relu(tf.add(feature3, pool)) 
    pool = helper.depthwiseConv2dRelu(pool,conv_s2_weights,conv_s2_biases)   
    featureMap.append(pool)

    pool = helper.upConv(pool, dconv_1_weights,dconv_1_biases, feature2.get_shape().as_list())
    pool = tf.nn.relu(tf.add(feature2, pool)) 
    pool = helper.depthwiseConv2dRelu(pool,conv_m2_weights,conv_m2_biases)   
    featureMap.append(pool)

    pool = helper.upConv(pool, dconv_0_weights,dconv_0_biases, feature1.get_shape().as_list())
    pool = tf.nn.relu(tf.add(feature1, pool)) 
    pool = helper.depthwiseConv2dRelu(pool,conv_l2_weights,conv_l2_biases,isDrop,keep_prop)    
    featureMap.append(pool)

    pool = helper.conv2dRelu(pool,conv_l3_weights,conv_l3_biases,isDrop,keep_prop)
  
    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
    reshape = tf.reshape(pool, [-1,LABEL_SIZE_C])
    if not train: reshape = tf.nn.softmax(reshape)
    return reshape,featureMap; 

def regullarizer():
    
    loss = tf.nn.l2_loss(conv_l0_weights) + tf.nn.l2_loss(conv_l2_weights) + tf.nn.l2_loss(conv_l3_weights)
    loss +=tf.nn.l2_loss(conv_m0_weights) + tf.nn.l2_loss(conv_m2_weights)
    loss +=tf.nn.l2_loss(conv_s0_weights) + tf.nn.l2_loss(conv_s2_weights)
    loss +=tf.nn.l2_loss(conv_t0_weights) + tf.nn.l2_loss(conv_t2_weights)
    loss +=tf.nn.l2_loss(conv_p0_weights) 
    loss +=tf.nn.l2_loss(dconv_0_weights) + tf.nn.l2_loss(dconv_1_weights)
    loss +=tf.nn.l2_loss(dconv_2_weights) + tf.nn.l2_loss(dconv_3_weights)
    loss +=tf.nn.l2_loss(dconv_4_weights) + tf.nn.l2_loss(dconv_5_weights)
    return loss
