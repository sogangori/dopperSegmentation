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
import Train_helper as helper
import Model_trimap_bn as model 

folder = "./IQ_COLOR/weights/"
hiddenImagePath = folder+"hidden/"
ImagePath1 = folder+"trimap"
ImagePath2 = folder+"unknown"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
EVAL_BATCH_SIZE = 10

def main(argv=None):        
      
  test_data, test_label = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  X = tf.placeholder(tf.float32, [None,test_data.shape[1],test_data.shape[2],test_data.shape[3]])
  Y = tf.placeholder(tf.int32, [None,test_label.shape[1],test_label.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
   
  trimap,ff = model.inference(X, IsTrain, Step)        
  argMax = tf.cast( tf.arg_max(trimap,3), tf.int32)  
  argMax_known = tf.cast(argMax > 0, tf.int32)
  argMax_trimap = tf.cast( tf.arg_max(trimap[:,:,:,0:2],3), tf.int32)
  known = tf.cast( argMax < 2, tf.float32)  
  unknown = tf.cast( argMax > 1, tf.float32)  
  background = tf.cast( argMax < 1, tf.float32)
  unknown_mean = tf.reduce_mean(unknown)  
  background_mean = tf.reduce_mean(background)  
  foreground_mean = 1.0 - unknown_mean - background_mean
  mean_iou_known = helper.getIoU(Y,argMax_known)  
  mean_iou_trimap = helper.getIoU(Y,argMax_trimap)
  with tf.Session() as sess:    
    tf.train.Saver().restore(sess, model.modelName)
    print("Model restored")
   
    start_time = time.time()
    feed_dict_test = {X: test_data, Y: test_label,IsTrain:False,Step:0}
    unkno,fore,back, iou_tri, iou_known = sess.run(
        [unknown_mean, foreground_mean,background_mean, mean_iou_trimap, mean_iou_known], feed_dict_test)
    elapsed_time = time.time() - start_time        
                
    print('%.0f ms, trimap(%.1f, %.1f, %.1f), IoU_tri(%g), iou_known:%g' % 
            (elapsed_time,back*100,fore*100,unkno*100,iou_tri, iou_known*100))  
    trimap_mask,unknown_mask = sess.run([trimap,unknown], feed_dict= feed_dict_test)    
    DataReader.SaveAsImage(unknown_mask, ImagePath2, trimap_mask.shape[0])    
    print ('trimap_mask',trimap_mask.shape)
    DataReader.SaveImage(trimap_mask,ImagePath1)


tf.app.run()


