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
import Model_bimap as model 
import Model_trimap_2x as modelTrimap 
folder = "./DAS_Unknown/weights/"
hiddenImagePath = folder+"hidden/"
ImagePath1 = folder+"bimap"
ImagePath2 = folder+"final"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
DATA_SIZE = 12 
BATCH_SIZE = np.int(DATA_SIZE) 

def getLossMSE(bimap, labels_node, trimap):   
     
    shape = tf.shape(bimap)
    label = tf.one_hot(labels_node,2)    
    error = tf.square(label - bimap)
    bimap_prob = tf.nn.softmax(bimap)
    unknown_idx = tf.cast(tf.arg_max(trimap,3) > 1, tf.float32)+0.1
    unknown_4d = tf.reshape(unknown_idx, [-1,shape[1],shape[2],1])    
    unknown_mask = tf.concat([unknown_4d,unknown_4d],3)    
    error_bimap = tf.multiply(error,unknown_mask)    
    return tf.reduce_mean(error_bimap)

def main(argv=None):        

  ensemble = model.ensemble  
  train_data, train_labels,test_data,test_label = DataReader.GetDataTrainTest(DATA_SIZE,1,ensemble);  
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  false_co = tf.constant(False)
  trimap = modelTrimap.inference(X, false_co, Step)
  bimap = model.inference(X, IsTrain, Step)
  argMax = tf.cast( tf.arg_max(bimap,3), tf.int32)  
  bimap_fore_idx = tf.cast(tf.arg_max(bimap,3), tf.float32)    
  trimap_argMax = tf.arg_max(trimap,3)
  trimap_unknown_idx = tf.cast(trimap_argMax > 1, tf.float32)
  trimap_back_idx = tf.cast(trimap_argMax < 1, tf.float32)    
  trimap_fore_idx = tf.ones_like(trimap_back_idx) - trimap_back_idx - trimap_unknown_idx   
  final_bimap = tf.cast( tf.add(bimap_fore_idx,trimap_fore_idx)-trimap_back_idx>0 , tf.int32)
  mean_iou = getIoU(Y,argMax)
  mean_iou_final = getIoU(Y,final_bimap)
 
  variable_bimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'bimap')
  variable_trimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'trimap')
  
  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    tf.global_variables_initializer().run()
    saver_trimap = tf.train.Saver(variable_trimap)  
    saver_bimap = tf.train.Saver(variable_bimap)  
    saver_trimap.restore(sess, modelTrimap.modelName)
      
    saver_bimap.restore(sess, model.modelName)
    print("Model restored")
    sess.run(tf.local_variables_initializer())  
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    test_offset = train_data.shape[3] - ensemble  #288 - 12    
    start_offsets = np.arange(test_offset)
      
    bimap_mask,final_bimap_mask,iou,iou_final = sess.run([bimap,final_bimap,mean_iou,mean_iou_final], feed_dict_test)
          
    print('IoU(%g, %g)' % (iou*100,iou_final*100)) 
    print ('bimap_mask',bimap_mask.shape)
    DataReader.SaveImage(bimap_mask[:,:,:,1],ImagePath1)
    DataReader.SaveImage(final_bimap_mask,ImagePath2)    
    

def getIoU(label,predict):
    label = tf.round(label)    
    predict = tf.round(predict)
    trn_labels = tf.reshape(label, [-1])
    logits=tf.reshape(predict, [-1])
    inter = tf.multiply(logits,trn_labels)
    union = tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels))
    iou = tf.reduce_sum(inter)/tf.reduce_sum(union)
    return tf.cast(iou,tf.float32)


tf.app.run()