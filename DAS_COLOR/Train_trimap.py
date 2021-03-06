﻿from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time
import numpy as np
from time import localtime, strftime
import sklearn.metrics as metrics
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from operator import or_
from DataReader import DataReader
import Model_trimap_2x as model 
#http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
folder = "./DAS_COLOR/weights/"
hiddenImagePath = folder+"hidden/"
predictImagePath = folder+"predict"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
EVAL_FREQUENCY = 10
AUGMENT = 4
DATA_SIZE = 12
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 1
isNewTrain = True      

def getLossMSE(trimap, labels_node):    
    label = tf.one_hot(labels_node,3)
    predict = trimap
    error = tf.square(label-predict)
    known = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)
    # = 1+tf.nn.softmax(trimap)[:,:,:,2]
    #known = tf.multiply(known,error_w)
    shape = tf.shape(known)
    known_re = tf.reshape(known, [-1,shape[1],shape[2],1])    
    known_prob = tf.reshape(1-tf.nn.softmax(trimap)[:,:,:,2],[-1,shape[1],shape[2],1])     
    #known_2d = tf.concat([known_re,known_re,tf.zeros_like(known_re)],3)
    known_2d = tf.concat([known_re,known_re,known_prob],3)
    error_bimap = tf.multiply(error,known_2d)
    return tf.reduce_mean(error),tf.reduce_mean(error_bimap)

def getLossMSE2(trimap, labels_node):    
    shape = tf.shape(trimap)
    label = tf.one_hot(labels_node,2)
    bimap = trimap[:,:,:,0:2]
    error = tf.square(label - bimap)
    known = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)
    
    known_re = tf.reshape(known, [-1,shape[1],shape[2],1])    
    known_2d = tf.concat([known_re,known_re],3)    
    error_bimap = tf.multiply(error,known_2d)
    
    return tf.reduce_mean(error_bimap)

def getLossMSE_penalty(trimap, labels_node):    
    shape = tf.shape(trimap)
    label = tf.one_hot(labels_node,2)
    bimap = trimap[:,:,:,0:2]
    error = tf.square(label - bimap)
    trimap_prob = tf.nn.softmax(trimap)
    known_idx = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)
    known_prob = known_idx * (1-trimap_prob[:,:,:,2])
    known_re = tf.reshape(known_prob, [-1,shape[1],shape[2],1])    
    weight = tf.concat([known_re,known_re],3)    
    error_bimap = tf.multiply(error,weight)    
    return tf.reduce_mean(error_bimap)

def main(argv=None):        

  ensemble = model.ensemble  
  train_data, train_labels,test_data,test_label = DataReader.GetDataTrainTest(DATA_SIZE,AUGMENT,ensemble);  
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  
  trimap = model.inference(X, IsTrain, Step)        
  argMax = tf.cast( tf.arg_max(trimap,3), tf.int32)  
  known = tf.cast( argMax < 2, tf.float32)  
  unknown = tf.cast( argMax > 1, tf.float32)  
  background = tf.cast( argMax < 1, tf.float32)
  unknown_mean = tf.reduce_mean(unknown)  
  background_mean = tf.reduce_mean(background)  
  foreground_mean = 1.0 - unknown_mean - background_mean
  
  Y_known = tf.multiply(tf.cast(Y, tf.float32), known)
  argMax_known = tf.multiply(tf.cast(argMax, tf.float32), known)
  mean_iou_known = getIoU(Y_known,argMax_known)
  mean_iou = getIoU(Y,argMax)
  entropy = getLossMSE_penalty(trimap, Y)    
  loss = entropy + 1e-8 * tf.nn.l2_loss(tf.nn.softmax(trimap)[:,:,:,2]) + 1e-5 * regularizer()    
  batch = tf.Variable(0)
  LearningRate = 0.01
  DecayRate = 0.999
  
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
  
  # Use simple momentum for the optimization.
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch) 

  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction=0.98
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    saver = tf.train.Saver()  
    if isNewTrain:
        tf.global_variables_initializer().run()
        print('Initialized!')
    else :        
        saver.restore(sess, model.modelName)
        print("Model restored")
    sess.run(tf.local_variables_initializer())  
    #summary_writer = tf.train.SummaryWriter(model.logName, sess.graph)
    #merged = tf.merge_all_summaries()
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    test_offset = train_data.shape[3] - ensemble  #288 - 12    
    start_offsets = np.arange(test_offset)
    for step in xrange(NUM_EPOCHS):
      model.step = step      
      np.random.shuffle(start_offsets)
      for iter in range((int)(test_offset/1)):
          batch_data = train_data[:,:,:,start_offsets[iter]:start_offsets[iter] + ensemble]          
          feed_dict = {X: batch_data, Y: train_labels, IsTrain:True,Step:step}      
          _,unkno,fore,back, l, iou,iou_known,lr = sess.run([optimizer,unknown_mean, foreground_mean,background_mean,entropy, mean_iou, mean_iou_known,learning_rate], feed_dict)
          
          if iter % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY           
            iou_test = sess.run(mean_iou, feed_dict_test)
        
            print('e%d,i%d,%.0fms,tri(%.0f,%.0f,%.0f),L:%.3f,IoU(%.0f,%.0f),Iou_k:%.1f,lr %.4f, %s' % 
                  (step, iter,takes,back*100,fore*100,unkno*100,l,iou*100,iou_test*100, iou_known*100,lr*100,now))   
          
            sys.stdout.flush()
            if lr==0 or l>20: 
                print ('lr l has problem  ',lr) 
                return
        
            this_sec = time.time()
            if this_sec - start_sec > 60 * 20 :
                start_sec = this_sec
                save_path = saver.save(sess, model.modelName)
                now = strftime("%H:%M:%S", localtime())
                print("Model Saved, time:%s" %(now))      
           
    if sess.run(learning_rate)>0: 
        save_path = saver.save(sess, model.modelName)
        print ('save_path', save_path)      
    
    trimap_mask,unknown_mask = sess.run([trimap,unknown], feed_dict= feed_dict_test)    
    DataReader.SaveAsImage(unknown_mask, predictImagePath, trimap_mask.shape[0])    
    print ('trimap_mask',trimap_mask.shape)
    DataReader.SaveImage(trimap_mask,predictImagePath)
    

def getIoU(label,predict):
    label = tf.round(label)    
    predict = tf.round(predict)
    trn_labels = tf.reshape(label, [-1])
    logits=tf.reshape(predict, [-1])
    inter = tf.multiply(logits,trn_labels)
    union = tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels))
    iou = tf.reduce_sum(inter)/tf.reduce_sum(union)
    return tf.cast(iou,tf.float32)

def getLossTrimap(trimap, labels_node):    
    trimap_shape = tf.shape(trimap)
    unknown = tf.cast( tf.arg_max(trimap,3) < 2, tf.int32)    
    unknown_reshape = tf.reshape(unknown, [-1])
    unknown_reshape_flt32 = tf.cast(unknown_reshape,tf.float32)
    trimap_serial =  tf.reshape(trimap, [-1, 3])
    trimap_serial = tf.nn.softmax(trimap_serial)
    label_reshape = tf.reshape(labels_node, [-1])    
    label_reshape = tf.multiply(label_reshape, unknown_reshape)
    label_reshape = tf.cast(label_reshape,tf.int32)
    hot = tf.cast( tf.one_hot(unknown_reshape,3),tf.float32)
    trimap_serial2 = tf.multiply(trimap_serial, hot)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = trimap_serial, labels = label_reshape)    
    entropy2 = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = trimap_serial2, labels = label_reshape)    
    print('trimap',trimap)
    print('entropy', entropy)
    print('unknown_reshape', unknown_reshape)
    return  tf.reduce_mean(entropy),tf.reduce_mean(entropy2) 

def regularizer():
    regula=0
    for var in tf.trainable_variables():         
        regula +=  tf.nn.l2_loss(var)
    return regula

tf.app.run()


