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
folder = "./DAS_Map/weights/"
hiddenImagePath = folder+"hidden/"
ImagePath1 = folder+"bimap"
ImagePath2 = folder+"final"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
DATA_SIZE = 1 

def main(argv=None):        

  ensemble = model.ensemble  
  train_data, train_labels,valid_data,valid_label, test_data,test_label = DataReader.GetData3(DATA_SIZE,1,ensemble);  

  #train_data = test_data = np.ones([1,4,4,12])
  #train_labels = test_label= np.ones([1,4,4])
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,test_data.shape[1],test_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,test_label.shape[1],test_label.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  false_co = tf.constant(False)
  
  bimap ,feature_list = model.inference(X, IsTrain, Step)
  argMax = tf.cast( tf.arg_max(bimap,3), tf.int32)      
  mean_iou = helper.getIoU(Y,argMax)  

  start_sec = start_time = time.time()
  with tf.Session() as sess:    
    tf.global_variables_initializer().run()    
    saver_bimap = tf.train.Saver()      
    
    saver_bimap.restore(sess, model.modelName)
    print("Model restored")
    sess.run(tf.local_variables_initializer())  
    feed_dict_train = {X: train_data[:,:,:,0:12], Y: train_labels, IsTrain :False,Step:0}
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    
    features, bimap_mask,mask,iou = sess.run([feature_list, bimap,argMax,mean_iou], feed_dict= feed_dict_train)            
    print('IoU: %.5f ' % (iou) )          
    #DataReader.SaveImage(bimap_mask[:,:,:,1],ImagePath1)
    DataReader.SaveImage(mask,ImagePath1)

tf.app.run()