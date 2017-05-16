from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time

import numpy
from time import localtime, strftime
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from operator import or_
from DataReader import DataReader
import Model_narrow_depthwise as model 

hiddenImagePath = "./Color/weights/hidden/"
predictImagePath = "./Color/weights/predict"
outImagePath = "./Color/weights/out"
inImagePath = "./Color/weights/in"

DataReader = DataReader()
EVAL_BATCH_SIZE = 100

def main(argv=None):        
  
  test_data, test_labels, compare_label = DataReader.GetDataCheck(EVAL_BATCH_SIZE,1, isTrain =  False);      
  print("test_data.shape", test_data.shape)
  
  train_size = test_data.shape[0]      
  
  test_data_node = tf.placeholder(tf.float32, shape=test_data.shape)
  test_labels_node = tf.placeholder(tf.int32, shape=test_labels.shape)
  prediction, feature_map = model.inference(test_data, False)    
  entropy = getLoss(prediction, test_labels_node)
    
  predict_reshape = tf.reshape(prediction[:,1],shape=test_labels.shape)
  accuracy = tf.reduce_mean(tf.cast(tf.equal(predict_reshape,test_labels), dtype= tf.float32))
   
  config=tf.ConfigProto()
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    saver = tf.train.Saver()      
    saver.restore(sess, model.modelName)
    print("Model restored")
    start = time.time()
    feed_dict = {test_data_node: test_data, test_labels_node: test_labels}    
    l,acc, predictions,featureMap = sess.run([entropy, accuracy, prediction,feature_map], feed_dict=feed_dict)
    
    inferenceTime = time.time() - start
    print('entropy %f, accuracy %f, inferenceTime %.2f' % (l,acc,inferenceTime))       
    
    DataReader.SaveAsImage(predictions[:,1], predictImagePath, EVAL_BATCH_SIZE, maxCount = 10)
    DataReader.SaveFeatureMap(featureMap , "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE, maxCount = 10)
    DataReader.SaveAsImage(compare_label, "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE, maxCount = 10)


def getLoss(prediction,labels_node):    
    shape = labels_node.get_shape().as_list()
    label_reshape = tf.reshape(labels_node, [-1])
    print ('prediction',prediction)
    print ('labels_node',labels_node)
    print ('labels_node_reshape',label_reshape)
    #A common use case is to have logits of shape [batch_size, num_classes] 
    #and labels of shape [batch_size]. But higher dimensions are supported.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = label_reshape)
    return  tf.reduce_mean(entropy)    


tf.app.run()


