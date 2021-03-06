﻿from __future__ import absolute_import
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

modelName = "./DAS_COLOR/weights/bn_ensemble.pd"
LABEL_SIZE_C = 2
ensemble = 12
ensemble_iq = 3
ensemble_hpf = ensemble - ensemble_iq + 1
depth0 = ensemble
pool_stride2 =[1, 2, 2, 1]
pool_stride3 =[1, 3, 3, 1]

conv_hpf_weights  = tf.get_variable("hpf", shape=[3, 3, ensemble_hpf, ensemble_hpf], initializer =tf.contrib.layers.xavier_initializer())
beta_hpf = tf.Variable(tf.constant(0.0, shape=[ensemble_hpf]), trainable=True)
gamma_hpf = tf.Variable(tf.constant(1.0, shape=[ensemble_hpf]), trainable=True)

conv_hpf_1_weights  = tf.get_variable("hpf_1", shape=[3, 3, ensemble_hpf, ensemble_iq], initializer =tf.contrib.layers.xavier_initializer())
beta_hpf_1 = tf.Variable(tf.constant(0.0, shape=[ensemble_iq]), trainable=True)
gamma_hpf_1 = tf.Variable(tf.constant(1.0, shape=[ensemble_iq]), trainable=True)

conv_hpf_2_weights  = tf.get_variable("hpf_2", shape=[3, 3, ensemble_iq, 1], initializer =tf.contrib.layers.xavier_initializer())
beta_hpf_2 = tf.Variable(tf.constant(0.0, shape=[1]), trainable=True)
gamma_hpf_2 = tf.Variable(tf.constant(1.0, shape=[1]), trainable=True)

conv_l0_weights  = tf.get_variable("w1", shape=[3, 3, ensemble_iq, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_l0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_l0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True)

conv_m0_weights = tf.get_variable("m0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_m0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_m0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True)

conv_s0_weights = tf.get_variable("s0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_s0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_s0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True)

conv_t0_weights = tf.get_variable("t0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_t0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_t0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True) 

conv_p0_weights = tf.get_variable("p0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_p0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_p0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True) 

conv_x0_weights = tf.get_variable("x0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_x0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_x0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True) 

conv_xx0_weights = tf.get_variable("xx0", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_xx0 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_xx0 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True) 

conv_x2_weights = tf.get_variable("x2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_x2 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_x2 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True) 

conv_p2_weights = tf.get_variable("p2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_p2 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_p2 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True)

conv_t2_weights = tf.get_variable("t2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_t2 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_t2 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True)

conv_s2_weights = tf.get_variable("s2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_s2 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_s2 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True)

conv_m2_weights = tf.get_variable("m2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_m2 = tf.Variable(tf.constant(0.0, shape=[depth0]),name='beta_m2', trainable=True)
gamma_m2 = tf.Variable(tf.constant(1.0, shape=[depth0]),name='gamma_m2', trainable=True) 

conv_l2_weights = tf.get_variable("l2", shape=[3, 3, depth0, depth0], initializer =tf.contrib.layers.xavier_initializer())
beta_l2 = tf.Variable(tf.constant(0.0, shape=[depth0]), trainable=True)
gamma_l2 = tf.Variable(tf.constant(1.0, shape=[depth0]), trainable=True) 

conv_l3_weights = tf.get_variable("l3", shape=[3, 3, depth0, LABEL_SIZE_C], initializer =tf.contrib.layers.xavier_initializer())
beta_l3 = tf.Variable(tf.constant(0.0, shape=[LABEL_SIZE_C]),trainable=True)
gamma_l3 = tf.Variable(tf.constant(1.0, shape=[LABEL_SIZE_C]), trainable=True) 

def Get(src):
    return src

def hpf(src, train):
    out = helper.conv2dBN_Relu(src,conv_hpf_weights,beta_hpf,gamma_hpf,train)
    out = tf.nn.max_pool(out,pool_stride2,strides=pool_stride2,padding='SAME')    
    out = helper.conv2dBN_Relu(out,conv_hpf_1_weights,beta_hpf_1,gamma_hpf_1,train)
    out = helper.conv2dBN_Relu(out,conv_hpf_2_weights,beta_hpf_2,gamma_hpf_2,train)
    return out

def inference(inData, train,step):
    helper.isDrop = train
    helper.keep_prop = 0.8 
    
    #inData = tf.abs(inData)
    ensembleList = []
    for i in range(ensemble - ensemble_hpf + 1):
        ensembleList.append( hpf(inData[:,:,:,i:i+ensemble_hpf], train))
    ensembleMap = ensembleList[0]
    for i in range(1, ensemble - ensemble_hpf+1):
        ensembleMap = tf.concat([ensembleMap ,ensembleList[i]],3)
    
    #if step%3==1: in2= tf.nn.avg_pool(inData,pool_stride2,strides=pool_stride2,padding='SAME')
    #elif step%3==2:in2= tf.nn.avg_pool(inData,pool_stride3,strides=pool_stride3,padding='SAME')
    feature1 = pool = helper.conv2dBN_Relu(ensembleMap ,conv_l0_weights,beta_l0,gamma_l0,train)

    #1/4
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature2 = pool = helper.conv2dBN_Relu(pool,conv_m0_weights,beta_m0,gamma_m0, train)   

    #1/8
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature3 = pool = helper.conv2dBN_Relu(pool,conv_s0_weights,beta_s0,gamma_s0, train)   

    #1/16
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature4 = pool = helper.conv2dBN_Relu(pool,conv_t0_weights,beta_t0,gamma_t0, train)    

    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature5 = pool = helper.conv2dBN_Relu(pool,conv_p0_weights,beta_p0,gamma_p0, train)    
    
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    feature6 = pool = helper.conv2dBN_Relu(pool,conv_x0_weights,beta_x0,gamma_x0, train)    

    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = helper.conv2dBN(pool,conv_xx0_weights,beta_xx0,gamma_xx0, train)    

    up_shape = feature6.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature6, pool))
    pool = helper.conv2dBN(pool,conv_x2_weights,beta_x2,gamma_x2, train)    
    
    up_shape = feature5.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature5, pool)) 
    pool = helper.conv2dBN(pool,conv_p2_weights,beta_p2,gamma_p2, train)    
    
    up_shape = feature4.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature4, pool)) 
    pool = helper.conv2dBN(pool,conv_t2_weights,beta_t2,gamma_t2, train)    
    
    up_shape = feature3.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature3, pool)) 
    pool = helper.conv2dBN(pool,conv_s2_weights,beta_s2,gamma_s2, train)    
    
    up_shape = feature2.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature2, pool)) 
    pool = helper.conv2dBN_Relu(pool,conv_m2_weights,beta_m2,gamma_m2, train)

    up_shape = feature1.get_shape().as_list()
    pool = helper.resize(pool, up_shape[1],up_shape[2])    
    pool = tf.nn.relu(tf.add(feature1, pool)) 
    pool = helper.conv2dBN_Relu(pool,conv_l2_weights,beta_l2,gamma_l2, train)

    pool = helper.conv2dBN_Relu(pool,conv_l3_weights,beta_l3,gamma_l3, train)
    input_shape = inData.get_shape().as_list()
    pool = helper.resize(pool,input_shape[1] ,input_shape[2])
    
    return pool; 