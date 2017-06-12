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
import Model_bimap as model 
import Model_trimap as modelTrimap
#http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
hiddenImagePath = "./IQ_COLOR/weights/hidden/"
ImagePath2 = "./IQ_COLOR/weights/final"

DataReader = DataReader()
EVAL_BATCH_SIZE = 10             

def main(argv=None):
  test_data, test_labels = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);    
  X = tf.placeholder(tf.float32, [None,test_data.shape[1],test_data.shape[2],test_data.shape[3]])
  Y = tf.placeholder(tf.int32, [None,test_labels.shape[1],test_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  
  false_co = tf.constant(False)
  trimap,ffM = modelTrimap.inference(X, false_co, Step)
  bimap,ffM = model.inference(X, IsTrain, Step)
  argMax = tf.cast( tf.arg_max(bimap,3), tf.int32)  
  bimap_fore_idx = tf.cast(tf.arg_max(bimap,3), tf.float32)    
  trimap_argMax = tf.arg_max(trimap,3)
  trimap_unknown_idx = tf.cast(trimap_argMax > 1, tf.float32)
  trimap_back_idx = tf.cast(trimap_argMax < 1, tf.float32)    
  trimap_fore_idx = tf.ones_like(trimap_back_idx) - trimap_back_idx - trimap_unknown_idx   
  final_bimap = tf.cast( tf.add(bimap_fore_idx,trimap_fore_idx)-trimap_back_idx>0 , tf.int32)
  mean_iou = helper.getIoU(Y,argMax)
  mean_iou_final = helper.getIoU(Y,final_bimap)
    
  variable_bimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'bimap')
  variable_trimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'trimap')
  start_sec = start_time = time.time()
  with tf.Session() as sess:    
    tf.global_variables_initializer().run()
    saver_trimap = tf.train.Saver(variable_trimap)  
    saver_bimap = tf.train.Saver(variable_bimap)  
    saver_trimap.restore(sess, modelTrimap.modelName)      
    saver_bimap.restore(sess, model.modelName)
    print("Model restored")
    sess.run(tf.local_variables_initializer())  
  
    feed_dict = {X: test_data, Y: test_labels,IsTrain:False,Step:0}
     
    predict, iou,iou_final = sess.run([final_bimap,mean_iou, mean_iou_final], feed_dict)    
    print('IoU:%g, IoU_final: %g)' % (iou,iou_final))   
    DataReader.SaveAsImage(predict, ImagePath2, EVAL_BATCH_SIZE)
    #DataReader.SaveFeatureMap(feature_map, "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE)


tf.app.run()


