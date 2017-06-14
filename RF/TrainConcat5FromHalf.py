
# tensorboard --logdir=/home/way/NVPACK/nvsample_workspace/python-mnist/hi/patch.pd
"""Simple, end-to-end, LeNet-5-like convolutional MNIST model example.

This should achieve a test error of 0.7%. Please keep this model as simple and
linear as possible, it is meant as a tutorial for simple convolutional models.
Run with --self_test on the command line to execute a short self-test.
"""
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
from DataReader import DataReader

modelName = "./DAS/weights/concat5fromHalf.pd"
logName ="./DAS/weights/logs_concat5fromHalf"

isDropout = not True
LABEL_SIZE_W = 128
LABEL_SIZE_C = 1
NUM_CHANNELS_In= 255

SEED = time.time()

keep_prop = 0.8

pool_stride1 =[1, 1, 1, 1]
pool_stride2 =[1, 2, 1, 1]
pool_stride22 =[1, 2, 2, 1]
pool_stride3 =[1, 3, 1, 1]
DataReader = DataReader()

depth0 = 16
depth1 = 32
depth2 = 48
depth3 = 64
depth4 = 80

conv_o0_weights = tf.Variable(tf.truncated_normal([3, 3, NUM_CHANNELS_In, depth0],stddev=0.01,seed=SEED))
conv_o0_biases = tf.Variable(tf.zeros([depth0]))


conv_l0_weights = tf.Variable(tf.truncated_normal([3, 3, depth0, depth1],stddev=0.01,seed=SEED))
conv_l0_biases = tf.Variable(tf.zeros([depth1]))
conv_l1_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, depth1],stddev=0.01,seed=SEED))
conv_l1_biases = tf.Variable(tf.zeros([depth1]))

conv_m0_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, depth2],stddev=0.1,seed=SEED))
conv_m0_biases = tf.Variable(tf.zeros([depth2]))
conv_m1_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, depth2],stddev=0.1,seed=SEED))
conv_m1_biases = tf.Variable(tf.zeros([depth2]))

conv_s0_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, depth3],stddev=0.1,seed=SEED))
conv_s0_biases = tf.Variable(tf.zeros([depth3]))
conv_s1_weights = tf.Variable(tf.truncated_normal([3, 3, depth3, depth3],stddev=0.1,seed=SEED))
conv_s1_biases = tf.Variable(tf.zeros([depth3]))

conv_roi0_weights = tf.Variable(tf.truncated_normal([3, 3, depth3, depth4],stddev=0.1,seed=SEED))
conv_roi0_biases = tf.Variable(tf.zeros([depth4]))
conv_roi1_weights = tf.Variable(tf.truncated_normal([3, 3, depth4, depth3],stddev=0.1,seed=SEED))
conv_roi1_biases = tf.Variable(tf.zeros([depth3]))

conv_s2_weights = tf.Variable(tf.truncated_normal([3, 3, depth3+depth3, depth3],stddev=0.1,seed=SEED))
conv_s2_biases = tf.Variable(tf.zeros([depth3]))
conv_s3_weights = tf.Variable(tf.truncated_normal([3, 3, depth3, depth2],stddev=0.1,seed=SEED))
conv_s3_biases = tf.Variable(tf.zeros([depth2]))

conv_m2_weights = tf.Variable(tf.truncated_normal([3, 3, depth2+depth2, depth2],stddev=0.1,seed=SEED))
conv_m2_biases = tf.Variable(tf.zeros([depth2]))
conv_m3_weights = tf.Variable(tf.truncated_normal([3, 3, depth2, depth1],stddev=0.1,seed=SEED))
conv_m3_biases = tf.Variable(tf.zeros([depth1]))

conv_l2_weights = tf.Variable(tf.truncated_normal([3, 3, depth1+depth1, depth1],stddev=0.1,seed=SEED))
conv_l2_biases = tf.Variable(tf.zeros([depth1]))
conv_l3_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, depth0],stddev=0.1,seed=SEED))
conv_l3_biases = tf.Variable(tf.zeros([depth0]))

conv_o2_weights = tf.Variable(tf.truncated_normal([3, 3, depth0+depth0, depth0],stddev=0.1,seed=SEED))
conv_o2_biases = tf.Variable(tf.zeros([depth0]))
conv_o3_weights = tf.Variable(tf.truncated_normal([3, 3, depth0, depth1],stddev=0.1,seed=SEED))
conv_o3_biases = tf.Variable(tf.zeros([depth1]))
conv_o4_weights = tf.Variable(tf.truncated_normal([3, 3, depth1, LABEL_SIZE_C],stddev=0.1,seed=SEED))
conv_o4_biases = tf.Variable(tf.zeros([LABEL_SIZE_C]))

def resize(src, dstH, dstW,interpol = 0):
    
    if interpol ==0:
        return tf.image.resize_nearest_neighbor(src, [dstH, dstW])
    elif interpol ==1:
        return tf.image.resize_bilinear(src, [dstH, dstW])
    else:
        return tf.image.resize_bicubic(src, [dstH, dstW])

def conv2d(src, weights, bias):
    src = tf.nn.dropout(src, 0.9,seed = time.time()) 
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    return tf.nn.bias_add(conv,bias)

def conv2dAct(src, weights, bias):
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    conv = tf.nn.bias_add(conv,bias)    
    return tf.nn.relu(conv)  

def inference(inData, train=False):
    #1/1  - 1/2 - 1/4 - 1/8 - 1/16
    #1024 - 512 - 256 - 128 - 64
    inData = tf.multiply(inData,1.0)
    #1/2    
    pool = conv2dAct(inData,conv_o0_weights,conv_o0_biases)    
    featureO = pool
    
    #1/4 
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')
    pool = conv2dAct(pool,conv_l0_weights,conv_l0_biases)
    pool = conv2dAct(pool,conv_l1_weights,conv_l1_biases)
    featureL = pool  
    
    #1/8
    pool = tf.nn.max_pool(pool,pool_stride2,strides=pool_stride2,padding='SAME')    
    pool = conv2dAct(pool,conv_m0_weights,conv_m0_biases)
    pool = conv2dAct(pool,conv_m1_weights,conv_m1_biases)
    featureM = pool   
    
    #1/16
    pool = tf.nn.max_pool(pool,pool_stride22,strides=pool_stride22,padding='SAME')
    pool = conv2dAct(pool,conv_s0_weights,conv_s0_biases)
    pool = conv2dAct(pool,conv_s1_weights,conv_s1_biases)
    featureS = pool
    
    #1/32
    pool = tf.nn.max_pool(pool,pool_stride22,strides=pool_stride22,padding='SAME')
    pool = conv2dAct(pool,conv_roi0_weights,conv_roi0_biases)
    pool = conv2dAct(pool,conv_roi1_weights,conv_roi1_biases)    

    #1/16
    up_shape = featureS.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.concat([featureS, pool],3)
    pool = conv2dAct(pool,conv_s2_weights,conv_s2_biases)
    pool = conv2dAct(pool,conv_s3_weights,conv_s3_biases)     
    
    #1/8
    up_shape = featureM.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.concat([featureM, pool],3)
    pool = conv2dAct(pool,conv_m2_weights,conv_m2_biases) 
    pool = conv2dAct(pool,conv_m3_weights,conv_m3_biases)
    
    #1/4
    up_shape = featureL.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.concat([featureL, pool], 3)
    pool = conv2dAct(pool,conv_l2_weights,conv_l2_biases) 
    pool = conv2dAct(pool,conv_l3_weights,conv_l3_biases)
    
    #1/2
    up_shape = featureO.get_shape().as_list()
    pool = resize(pool, up_shape[1],up_shape[2],interpol = 1)    
    pool = tf.concat([featureO, pool],3)
    pool = conv2dAct(pool,conv_o2_weights,conv_o2_biases)
    pool = conv2dAct(pool,conv_o3_weights,conv_o3_biases)
    pool = conv2d(pool,conv_o4_weights,conv_o4_biases)
    pool = tf.nn.tanh(pool)
        
    #1/1
    
    pool_shape = pool.get_shape().as_list()
    reshape = tf.reshape(pool, [pool_shape[0], -1, LABEL_SIZE_W])
      
    return reshape; 

def getLoss(prediction,labels_node):    
    loss = tf.reduce_mean(tf.square( labels_node - prediction))    
    return loss

def regullarizer():

    regularizers = tf.nn.l2_loss(conv_o0_weights)  + tf.nn.l2_loss(conv_o2_weights)+ tf.nn.l2_loss(conv_o3_weights)+ tf.nn.l2_loss(conv_o4_weights)
    regularizers += tf.nn.l2_loss(conv_l0_weights) + tf.nn.l2_loss(conv_l1_weights) + tf.nn.l2_loss(conv_l2_weights) + tf.nn.l2_loss(conv_l3_weights)
    regularizers += tf.nn.l2_loss(conv_m0_weights) + tf.nn.l2_loss(conv_m2_weights) + tf.nn.l2_loss(conv_m3_weights)
    regularizers += tf.nn.l2_loss(conv_s0_weights) + tf.nn.l2_loss(conv_s2_weights) + tf.nn.l2_loss(conv_s3_weights)    
    regularizers += tf.nn.l2_loss(conv_roi0_weights) + tf.nn.l2_loss(conv_roi1_weights)
    
    return regularizers

def GetTrainData(isTrain = True,isRotate = False,count = 1,trainH = 128):    
    
    #if isRotate and isTrain:  trainingSet,trainingOut = DataReader.GetTrainDataToTensorflowRotateLR(count,LABEL_SIZE_H,LABEL_SIZE_W,LABEL_SIZE_C,isTrain); 
    #else : 
    trainingSet,trainingOut = DataReader.GetTrainDataToTensorflow(count,isTrain, trainH);
        
    return [trainingSet, trainingOut]

def SavePredictAsImage(src, path):    
    DataReader.SaveAsImage(src, path, src.shape[0])
    
def SavePredictAsImageByChannel(src, path):
    DataReader.SaveAsImageByChannel(src, path, src.shape[0])
    
def Accuracy(out ,predict):
    
    return DataReader.SNR(out ,predict)
    
