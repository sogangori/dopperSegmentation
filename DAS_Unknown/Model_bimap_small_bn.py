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

modelName = "./DAS_Unknown/weights/bimap_small_bn.pd"
LABEL_SIZE_C = 2
ensemble= 12
depth0 = ensemble
pool_stride2 =[1, 2, 2, 1]

with tf.variable_scope('bimap'):
    conv_l0_weights  = tf.get_variable("w1", shape=[3, 3, ensemble, depth0*2], initializer =tf.contrib.layers.xavier_initializer())
    beta_l0 = tf.Variable(tf.constant(0.0, shape=[depth0*2]))
    gamma_l0 = tf.Variable(tf.constant(1.0, shape=[depth0*2]))
    
    conv_m0_weights = tf.get_variable("m0", shape=[3, 3, depth0*2, depth0*4], initializer =tf.contrib.layers.xavier_initializer())
    beta_m0 = tf.Variable(tf.constant(0.0, shape=[depth0*4]))
    gamma_m0 = tf.Variable(tf.constant(1.0, shape=[depth0*4]))
    
    conv_s0_weights = tf.get_variable("s0", shape=[3, 3, depth0*4, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
    beta_s0 = tf.Variable(tf.constant(0.0, shape=[LABEL_SIZE_C]))    
    gamma_s0 = tf.Variable(tf.constant(1.0, shape=[LABEL_SIZE_C]))    

def inference(inData, train,step):
    helper.isDrop = train
    helper.isTrain = train
    helper.keep_prop = 0.99
        
    #0    
    pool = helper.conv2dBN_Relu(inData,conv_l0_weights,beta_l0,gamma_l0)
    
    #1
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN_Relu(pool,conv_m0_weights,beta_m0,gamma_m0)

    #2
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN_Relu(pool,conv_s0_weights,beta_s0,gamma_s0) 

    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
        
    return pool; 