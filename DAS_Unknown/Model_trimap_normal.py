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

modelName = "./DAS_Unknown/weights/nromalCH.pd"
LABEL_SIZE_C = 3
ensemble= 12
depth0 = ensemble
pool_stride2 =[1, 2, 2, 1]

with tf.variable_scope('trimap'):

    beta_in = tf.constant(0.0, shape=[ensemble])
    gamma_in = tf.constant(1.0, shape=[ensemble])

    conv_l0_weights = tf.get_variable("w1", shape=[1, 1, ensemble, depth0*2], initializer =tf.contrib.layers.xavier_initializer())
    beta_l0 = tf.Variable(tf.constant(0.0, shape=[depth0*2]))
    gamma_l0 = tf.Variable(tf.constant(1.0, shape=[depth0*2]))

    conv_m0_weights = tf.get_variable("m0", shape=[3, 3, depth0*2, depth0*3], initializer =tf.contrib.layers.xavier_initializer())
    beta_m0 = tf.Variable(tf.constant(0.0, shape=[depth0*3]))
    gamma_m0 = tf.Variable(tf.constant(1.0, shape=[depth0*3]))

    conv_s0_weights = tf.get_variable("s0", shape=[3, 3, depth0*3, depth0*4], initializer =tf.contrib.layers.xavier_initializer())
    beta_s0 = tf.Variable(tf.constant(0.0, shape=[depth0*4]))
    gamma_s0 = tf.Variable(tf.constant(1.0, shape=[depth0*4]))

    conv_t0_weights = tf.get_variable("t0", shape=[3, 3, depth0*4, depth0*5], initializer =tf.contrib.layers.xavier_initializer())
    beta_t0 = tf.Variable(tf.constant(0.0, shape=[depth0*5]))
    gamma_t0 = tf.Variable(tf.constant(1.0, shape=[depth0*5])) 

    conv_p0_weights = tf.get_variable("p0", shape=[3, 3, depth0*5, depth0*6], initializer =tf.contrib.layers.xavier_initializer())
    beta_p0 = tf.Variable(tf.constant(0.0, shape=[depth0*6]))
    gamma_p0 = tf.Variable(tf.constant(1.0, shape=[depth0*6])) 

    conv_x0_weights = tf.get_variable("x0", shape=[3, 3, depth0*6, depth0*7], initializer =tf.contrib.layers.xavier_initializer())
    beta_x0 = tf.Variable(tf.constant(0.0, shape=[depth0*7]))
    gamma_x0 = tf.Variable(tf.constant(1.0, shape=[depth0*7])) 

    conv_xx0_weights = tf.get_variable("xx0", shape=[3, 3, depth0*7, depth0*6], initializer =tf.contrib.layers.xavier_initializer())
    beta_xx0 = tf.Variable(tf.constant(0.0, shape=[depth0*6]))
    gamma_xx0 = tf.Variable(tf.constant(1.0, shape=[depth0*6])) 

    conv_x2_weights = tf.get_variable("x2", shape=[3, 3, depth0*6, depth0*6], initializer =tf.contrib.layers.xavier_initializer())
    beta_x2 = tf.Variable(tf.constant(0.0, shape=[depth0*6]))
    gamma_x2 = tf.Variable(tf.constant(1.0, shape=[depth0*6])) 

    conv_p2_weights = tf.get_variable("p2", shape=[3, 3, depth0*6, depth0*5], initializer =tf.contrib.layers.xavier_initializer())
    beta_p2 = tf.Variable(tf.constant(0.0, shape=[depth0*5]))
    gamma_p2 = tf.Variable(tf.constant(1.0, shape=[depth0*5]))

    conv_t2_weights = tf.get_variable("t2", shape=[3, 3, depth0*5, depth0*4], initializer =tf.contrib.layers.xavier_initializer())
    beta_t2 = tf.Variable(tf.constant(0.0, shape=[depth0*4]))
    gamma_t2 = tf.Variable(tf.constant(1.0, shape=[depth0*4]))

    conv_s2_weights = tf.get_variable("s2", shape=[3, 3, depth0*4, depth0*3], initializer =tf.contrib.layers.xavier_initializer())
    beta_s2 = tf.Variable(tf.constant(0.0, shape=[depth0*3]))
    gamma_s2 = tf.Variable(tf.constant(1.0, shape=[depth0*3]))

    conv_m2_weights = tf.get_variable("m2", shape=[3, 3, depth0*3, depth0*2], initializer =tf.contrib.layers.xavier_initializer())
    beta_m2 = tf.Variable(tf.constant(0.0, shape=[depth0*2]),name='beta_m2')
    gamma_m2 = tf.Variable(tf.constant(1.0, shape=[depth0*2]),name='gamma_m2') 

    conv_l2_weights = tf.get_variable("l2", shape=[3, 3, depth0*2, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_l2 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_l2 = tf.Variable(tf.constant(1.0, shape=[depth0])) 

    conv_l3_weights = tf.get_variable("l3", shape=[3, 3, depth0, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
    beta_l3 = tf.Variable(tf.constant(0.0, shape=[LABEL_SIZE_C]))
    gamma_l3 = tf.Variable(tf.constant(1.0, shape=[LABEL_SIZE_C])) 


def inference(inData, train,step):
    helper.isDrop = train
    helper.isTrain = train
    helper.keep_prop = 0.7
       
    inData = helper.StdNormalizeAll(inData,beta_in,gamma_in)
    
    inData = helper.Gaussian_noise_Add(inData, 0.1, 0.1)
    #0
    pool = helper.conv2dBN_Relu(inData,conv_l0_weights,beta_l0,gamma_l0)

    #1
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature0 = pool = helper.conv2dBN_Relu(pool,conv_m0_weights,beta_m0,gamma_m0)   

    #2
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature1 = pool = helper.conv2dBN_Relu(pool,conv_s0_weights,beta_s0,gamma_s0)   

    #3
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature2 = pool = helper.conv2dBN_Relu(pool,conv_t0_weights,beta_t0,gamma_t0)    

    #4
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = helper.conv2dBN_Relu(pool,conv_p0_weights,beta_p0,gamma_p0)    
    
    #5
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN_Relu(pool,conv_x0_weights,beta_x0,gamma_x0)    
    #6
    pool = helper.conv2dBN(pool,conv_xx0_weights,beta_xx0,gamma_xx0)    
    #7
    up_shape = feature3.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature3, pool)) 
    pool = helper.conv2dBN_Relu(pool,conv_x2_weights,beta_x2,gamma_x2)        
    #8
    pool = helper.conv2dBN(pool,conv_p2_weights,beta_p2,gamma_p2)   
    #9
    up_shape = feature2.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature2, pool)) 
    pool = helper.conv2dBN(pool,conv_t2_weights,beta_t2,gamma_t2)   
    #10
    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature1, pool)) 
    pool = helper.conv2dBN(pool,conv_s2_weights,beta_s2,gamma_s2)        
    #11
    up_shape = feature0.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature0, pool)) 
    #12
    pool = helper.conv2dBN_Relu(pool,conv_m2_weights,beta_m2,gamma_m2)
    #13
    pool = helper.conv2dBN_Relu(pool,conv_l2_weights,beta_l2,gamma_l2)
    #14
    pool = helper.conv2dBN_Relu(pool,conv_l3_weights,beta_l3,gamma_l3)
    #15
    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
        
    return pool; 