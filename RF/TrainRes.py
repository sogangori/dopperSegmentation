
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

modelName = "./DAS/weights/res.pd"
logName ="./DAS/weights/logs_res"

#$ tensorboard --logdir=/home/digits/workspace_git/cmode_tensorflow/train0/weights/logs_color1
#browser  http://0.0.0.0:6006

isDropout = not True
LABEL_SIZE_W = 128
LABEL_SIZE_C = 1
NUM_CHANNELS_In= 255

SEED = time.time()

keep_prop = 0.9 

pool_stride1 =[1, 1, 1, 1]
pool_stride2 =[1, 2, 1, 1]
pool_stride22 =[1, 2, 2, 1]

depth0 = 32
depth1 = 32
depth2 = 32
depth3 = 32

conv_l0_weights = tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS_In, depth0],stddev=0.1,seed=SEED))
conv_l0_biases = tf.Variable(tf.zeros([depth0]))

conv_m0_weights = tf.Variable(tf.truncated_normal([3, 3, depth0, depth1],stddev=0.1,seed=SEED))
conv_m0_biases = tf.Variable(tf.zeros([depth1]))

conv_s0_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, depth2],stddev=0.1,seed=SEED))
conv_s0_biases = tf.Variable(tf.zeros([depth2]))

conv_roi0_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, depth3],stddev=0.1,seed=SEED))
conv_roi0_biases = tf.Variable(tf.zeros([depth3]))
conv_roi1_weights = tf.Variable(tf.truncated_normal([3, 3, depth3, depth2],stddev=0.1,seed=SEED))
conv_roi1_biases = tf.Variable(tf.zeros([depth2]))

conv_s2_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, depth2],stddev=0.1,seed=SEED))
conv_s2_biases = tf.Variable(tf.zeros([depth2]))
conv_s3_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, depth1],stddev=0.1,seed=SEED))
conv_s3_biases = tf.Variable(tf.zeros([depth1]))

conv_m2_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, depth1],stddev=0.1,seed=SEED))
conv_m2_biases = tf.Variable(tf.zeros([depth1]))
conv_m3_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, depth0],stddev=0.1,seed=SEED))
conv_m3_biases = tf.Variable(tf.zeros([depth0]))

conv_l2_weights = tf.Variable(tf.truncated_normal([3, 3, depth0, depth2],stddev=0.1,seed=SEED))
conv_l2_biases = tf.Variable(tf.zeros([depth2]))
conv_l3_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, LABEL_SIZE_C],stddev=0.1,seed=SEED))
conv_l3_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))
conv_l4_weights = tf.Variable(tf.truncated_normal([3, 3, LABEL_SIZE_C, LABEL_SIZE_C],stddev=0.1,seed=SEED))
conv_l4_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))

def resize(src, dstH, dstW,interpol = 1):
    
    if interpol ==0: return tf.image.resize_nearest_neighbor(src, [dstH, dstW])
    elif interpol ==1: return tf.image.resize_bilinear(src, [dstH, dstW])
    else: return tf.image.resize_bicubic(src, [dstH, dstW])

def conv2d(src, weights, bias):
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    return tf.nn.bias_add(conv,bias)

def conv2dRelu(src, weights, bias, isDrop= False):
    if isDrop: inData = tf.nn.dropout(inData, 0.9, seed=SEED)
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    conv = tf.nn.bias_add(conv,bias)    
    return tf.nn.relu(conv) 

def inference(inData, train=False):
    isDrop = train and isDropout
    
    #1/2
    in1 =  featureO = tf.multiply(inData ,1.0)
    #in1 = tf.nn.avg_pool(featureO,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = conv2dRelu(in1,conv_l0_weights,conv_l0_biases,isDrop)
    featureL = pool  
    
    #1/4
    in2 = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = conv2dRelu(in2,conv_m0_weights,conv_m0_biases,isDrop)
    featureM = pool   
    
    #1/8
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')
    pool = conv2dRelu(pool,conv_s0_weights,conv_s0_biases,isDrop)
    featureS = pool
    
    #1/16
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')
    pool = conv2dRelu(pool,conv_roi0_weights,conv_roi0_biases,isDrop)
    pool = conv2d(pool,conv_roi1_weights,conv_roi1_biases)

    #1/8
    up_shape = featureS.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.add(featureS, pool)
    pool = conv2dRelu(pool,conv_s2_weights,conv_s2_biases,isDrop)
    pool = conv2d(pool,conv_s3_weights,conv_s3_biases)
    
    #1/4
    up_shape = featureM.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.add(featureM, pool)
    pool = conv2dRelu(pool,conv_m2_weights,conv_m2_biases,isDrop)
    pool = conv2d(pool,conv_m3_weights,conv_m3_biases)
    
    #1/2
    up_shape = featureL.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.add(featureL, pool) 
    pool = conv2dRelu(pool,conv_l2_weights,conv_l2_biases,isDrop)
    
    #1/1
    #up_shape = featureO.get_shape().as_list()
    #pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)  
    pool = conv2d(pool,conv_l3_weights,conv_l3_biases)
    pool = tf.nn.tanh(pool)
    pool = conv2d(pool,conv_l4_weights,conv_l4_biases)
    pool = tf.nn.tanh(pool)
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], -1, LABEL_SIZE_W])
      
    return reshape; 

def regullarizer():
    
    regularizers = tf.nn.l2_loss(conv_l0_weights)+ tf.nn.l2_loss(conv_l2_weights) + tf.nn.l2_loss(conv_l3_weights)+ tf.nn.l2_loss(conv_l4_weights)
    regularizers += tf.nn.l2_loss(conv_m0_weights) + tf.nn.l2_loss(conv_m2_weights) + tf.nn.l2_loss(conv_m3_weights)
    regularizers += tf.nn.l2_loss(conv_s0_weights) + tf.nn.l2_loss(conv_s2_weights) + tf.nn.l2_loss(conv_s3_weights)    
    regularizers += tf.nn.l2_loss(conv_roi0_weights) + tf.nn.l2_loss(conv_roi1_weights)
    
    return regularizers
