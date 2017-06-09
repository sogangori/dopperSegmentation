from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import sys
import numpy
from six.moves import urllib
from six.moves import xrange 
import tensorflow as tf
import Model_helper2 as helper

modelName = "./Color/weights/depthwise3NOAVG.pd"
#logName ="./Color/weights/logs_deconv3"

LABEL_SIZE_C = 2
NUM_CHANNELS_In= 3
pool_stride2 =[1, 2, 2, 1]

#depth 2, Aug x3~x4 : 83%, 80% loss 0.041x 
#depth 3, deconv : 85%, 82% loss 0.118x
#depthwise 3, deconv : 85%, 84% loss 0.119x, Result Good! 
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

conv_xx0_weights = tf.get_variable("xx0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
conv_xx0_biases = tf.Variable(tf.zeros([depth0]))

conv_x2_weights = tf.get_variable("x2", shape=[3, 3, depth0, 1], initializer =tf.contrib.layers.xavier_initializer())
conv_x2_biases = tf.Variable(tf.zeros([depth0]))

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
dconv_final_weights = tf.get_variable("d6", shape=[3, 3, LABEL_SIZE_C, depth0], initializer =tf.contrib.layers.xavier_initializer())
dconv_final_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))

def inference(inData, train,step):
    helper.isDrop = train
    helper.keep_prop = 0.65   
        
    #1/2
        
    in2 = tf.nn.avg_pool(inData,[1, 2, 2, 1],strides=pool_stride2,padding='SAME')    
    #in2 = inData 
    feature1 = pool = helper.conv2dRelu(in2,conv_l0_weights,conv_l0_biases)
    
    #1/4    
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature2 = pool = helper.conv2dRelu(pool,conv_m0_weights,conv_m0_biases)

    #1/8
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = helper.conv2dRelu(pool,conv_s0_weights,conv_s0_biases)   

    #1/16
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature4 = pool = helper.conv2dRelu(pool,conv_t0_weights,conv_t0_biases)   

    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature5 = pool = helper.conv2dRelu(pool,conv_p0_weights,conv_p0_biases)   
    
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature6 = pool = helper.conv2dRelu(pool,conv_x0_weights,conv_x0_biases)   
    
    pool= tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dRelu(pool,conv_xx0_weights,conv_xx0_biases)
    
    pool = helper.upConv(pool, dconv_5_weights,dconv_5_biases, tf.shape(feature6))
    pool = tf.nn.relu(tf.add(feature6, pool) )
    pool = helper.depthwiseConv2dRelu(pool,conv_x2_weights,conv_x2_biases)   
    
    
    pool = helper.upConv(pool, dconv_4_weights,dconv_4_biases, tf.shape(feature5))
    pool = tf.nn.relu(tf.add(feature5, pool) )
    pool = helper.depthwiseConv2dRelu(pool,conv_p2_weights,conv_p2_biases)   
    
 
    pool = helper.upConv(pool, dconv_3_weights,dconv_3_biases, tf.shape(feature4))
    pool = tf.nn.relu(tf.add(feature4, pool) )
    pool = helper.depthwiseConv2dRelu(pool,conv_t2_weights,conv_t2_biases)   
    

    pool = helper.upConv(pool, dconv_2_weights,dconv_2_biases, tf.shape(feature3))
    pool = tf.nn.relu(tf.add(feature3, pool)) 
    pool = helper.depthwiseConv2dRelu(pool,conv_s2_weights,conv_s2_biases)   
    

    pool = helper.upConv(pool, dconv_1_weights,dconv_1_biases, tf.shape(feature2))
    pool = tf.nn.relu(tf.add(feature2, pool)) 
    pool = helper.depthwiseConv2dRelu(pool,conv_m2_weights,conv_m2_biases)  

    pool = helper.upConv(pool, dconv_0_weights,dconv_0_biases, tf.shape(feature1))
    pool = tf.nn.relu(tf.add(feature1, pool)) 
    pool = helper.depthwiseConv2dRelu(pool,conv_l2_weights,conv_l2_biases)        
    
    shape = tf.shape(inData)
    print ('----------------------shape',shape)
    shape = [shape[0],shape[1],shape[2],2]
    pool = helper.upConvRelu(pool,dconv_final_weights,dconv_final_biases, shape)   
 
    return pool; 

