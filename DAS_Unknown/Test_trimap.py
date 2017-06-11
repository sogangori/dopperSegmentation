from __future__ import absolute_import
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
folder = "./DAS_Unknown/weights/"

predictImagePath = folder+"predict"
predictImagePath2 = folder+"tri_bimap"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
AUGMENT = 1
DATA_SIZE = 12

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
  argMax_trimap = tf.cast( tf.arg_max(trimap[:,:,:,0:2],3), tf.int32)
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
  mean_iou_trimap = getIoU(Y,argMax_trimap)
  entropy = getLossMSE_penalty(trimap, Y)    
  loss = entropy + 1e-8 * tf.nn.l2_loss(tf.nn.softmax(trimap)[:,:,:,2]) + 1e-5 * regularizer()    
  batch = tf.Variable(0) 

  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction=0.98
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    saver = tf.train.Saver()  
    
    saver.restore(sess, model.modelName)
    print("Model restored")
    sess.run(tf.local_variables_initializer())  
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    test_offset = train_data.shape[3] - ensemble  #288 - 12    
    start_offsets = np.arange(test_offset)   
    
    start_time = time.time()
    unkno,fore,back, l,iou_tri, iou,iou_known = sess.run([unknown_mean, foreground_mean,background_mean,entropy, mean_iou_trimap,mean_iou, mean_iou_known], feed_dict_test)
    elapsed_time = time.time() - start_time        
                
    print('%.0f ms, trimap(%.1f, %.1f, %.1f), L:%g, IoU_tri(%g), IoU(%g), Iou_k:%g' % 
            (elapsed_time,back*100,fore*100,unkno*100,l,iou_tri,iou*100, iou_known*100))   
          
    sys.stdout.flush()
    
    tri_bimap, trimap_mask,unknown_mask = sess.run([argMax_trimap,trimap,unknown], feed_dict= feed_dict_test)    
    DataReader.SaveAsImage(unknown_mask, predictImagePath, trimap_mask.shape[0])    
    print ('trimap_mask',trimap_mask.shape)
    DataReader.SaveImage(trimap_mask,predictImagePath)
    DataReader.SaveImage(tri_bimap,predictImagePath2)
    

def getIoU(label,predict):
    label = tf.round(label)    
    predict = tf.round(predict)
    trn_labels = tf.reshape(label, [-1])
    logits = tf.reshape(predict, [-1])
    inter = tf.multiply(logits,trn_labels)
    union = tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels))
    iou = tf.reduce_sum(inter)/tf.reduce_sum(union)
    return tf.cast(iou,tf.float32)

def regularizer():
    regula=0
    for var in tf.trainable_variables():         
        regula +=  tf.nn.l2_loss(var)
    return regula

tf.app.run()


