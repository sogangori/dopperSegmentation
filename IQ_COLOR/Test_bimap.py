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
#http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
hiddenImagePath = "./IQ_COLOR/weights/hidden/"
predictImagePath = "./IQ_COLOR/weights/predict"

DataReader = DataReader()
EVAL_BATCH_SIZE = 10
               
#82 43%, bn: 83 39
def main(argv=None):
  test_data, test_labels = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  
  X = tf.placeholder(tf.float32, [None,test_data.shape[1],test_data.shape[2],test_data.shape[3]])
  Y = tf.placeholder(tf.int32, [None,test_labels.shape[1],test_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  
  prediction, feature_map = model.inference(X, IsTrain, Step)    
  argMax = tf.cast( tf.arg_max(prediction,3), tf.int32)
  accuracy = tf.contrib.metrics.accuracy(argMax,Y)     
  mean_iou = helper.getIoU(Y,argMax)
  entropy = getLoss(prediction, Y)
  loss_iou = 1 - mean_iou   
    
  with tf.Session() as sess:    
    tf.train.Saver().restore(sess, model.modelName)
    print("Model restored")      
    feed_dict = {X: test_data, Y: test_labels,IsTrain:False,Step:0}
    predict,feature_map_ ,acc, iou = sess.run([prediction, feature_map,accuracy,mean_iou], feed_dict)        
    print('IoU (%g)' % (iou))  
            
    DataReader.SaveAsImage(predict[:,:,:,1], predictImagePath, EVAL_BATCH_SIZE)
    #DataReader.SaveFeatureMap(feature_map, "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE)

def getLoss(prediction,labels_node):    
    prediction =  tf.reshape(prediction, [-1, model.LABEL_SIZE_C])
    shape = labels_node.get_shape().as_list()
    label_reshape = tf.reshape(labels_node, [-1])
    print ('prediction',prediction)
    print ('labels_node',labels_node)
    print ('labels_node_reshape',label_reshape)
    #A common use case is to have logits of shape [batch_size, num_classes] 
    #and labels of shape [batch_size]. But higher dimensions are supported.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = label_reshape)
    #labelResionError = (entropy * tf.cast(label_reshape,tf.float32)) * 0.2
    return  tf.reduce_mean(entropy)    

tf.app.run()


