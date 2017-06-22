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

folder = "./ROI/weights/"
modelName = folder+"roi.pd"
LABEL_SIZE_C = 2
ensemble= 1
depth0 = ensemble
pool_stride2 =[1, 2, 3, 1]
#[1, 2, 2, 1] : (416,256) , (208,128), (104,64), (57,32) (26,16)
#[1, 2, 3, 1] : (416,256) , (208,86), (104,29), (52,10) (26,4)
with tf.variable_scope('roi'):
    conv_l0_weights  = tf.get_variable("w1", shape=[3, 3, ensemble, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_l0 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_l0 = tf.Variable(tf.constant(1.0, shape=[depth0]))

    conv_m0_weights = tf.get_variable("m0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_m0 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_m0 = tf.Variable(tf.constant(1.0, shape=[depth0]))

    conv_s0_weights = tf.get_variable("s0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_s0 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_s0 = tf.Variable(tf.constant(1.0, shape=[depth0]))

    conv_t0_weights = tf.get_variable("t0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_t0 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_t0 = tf.Variable(tf.constant(1.0, shape=[depth0])) 

    conv_p0_weights = tf.get_variable("p0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_p0 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_p0 = tf.Variable(tf.constant(1.0, shape=[depth0])) 

    conv_x0_weights = tf.get_variable("x0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
    beta_x0 = tf.Variable(tf.constant(0.0, shape=[depth0]))
    gamma_x0 = tf.Variable(tf.constant(1.0, shape=[depth0]))
    fc_in = 26*4*ensemble  #2*8 -> 6*16
    fc_weights =  tf.get_variable("fc", shape=[fc_in, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
    fc_bias =  tf.Variable(tf.constant(0.0, shape=[LABEL_SIZE_C]))

def inference(inData, train,step):
    helper.isDrop = train
    helper.isTrain = train
    helper.keep_prop = 0.7

    in_shape = tf.shape(inData)
    inData = tf.reshape(inData, [-1, in_shape[1], in_shape[2],1])
    
    pool = tf.nn.avg_pool(inData,pool_stride2,strides=pool_stride2,padding='SAME')       
    pool = helper.Gaussian_noise_Add(pool, 0.1, 0.1)
    #0
    pool = helper.conv2dBN_Relu(pool,conv_l0_weights,beta_l0,gamma_l0)
    pool = helper.conv2dBN_Relu(pool,conv_m0_weights,beta_m0,gamma_m0)   

    #1
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN_Relu(pool,conv_s0_weights,beta_s0,gamma_s0)   
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN_Relu(pool,conv_t0_weights,beta_t0,gamma_t0)    

    #2
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN_Relu(pool,conv_p0_weights,beta_p0,gamma_p0)       
    pool = helper.conv2dBN_Relu(pool,conv_x0_weights,beta_x0,gamma_x0)

    pool = tf.reshape(pool, [-1, fc_in])    
    pool = tf.matmul(pool, fc_weights)
    pool = tf.add(pool, fc_bias)
    pool = tf.nn.sigmoid(pool)
        
    return pool; 