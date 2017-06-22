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
import Model_smap as model 
folder = "./DAS_Map/weights/"
hiddenImagePath = folder+"hidden/"
ImagePath1 = folder+"bimap"
ImagePath2 = folder+"final"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
EVAL_FREQUENCY = 10
AUGMENT = 1
DATA_SIZE = 12
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 100
isNewTrain = not True      

def main(argv=None):        

  ensemble = model.ensemble  
  train_data, train_labels,valid_data,valid_label, test_data,test_label = DataReader.GetData3(DATA_SIZE,AUGMENT,ensemble);  
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  false_co = tf.constant(False)
  
  bimap = model.inference(X, IsTrain, Step)

  src_shape = Y.get_shape().as_list()
  dst_shape = bimap.get_shape().as_list()
  Y_4d = tf.reshape(Y, [-1,src_shape[1] ,src_shape[2],1])
  Y_s = helper.resize(Y_4d,dst_shape[1] ,dst_shape[2], interpol = 1)
  Y_s = tf.round(Y_s)
  Y_s = tf.reshape(Y_s, [-1,dst_shape[1] ,dst_shape[2]])
  Y_s = tf.cast(Y_s, tf.int32)
  argMax = tf.cast( tf.arg_max(bimap,3), tf.int32)      
  mean_iou = helper.getIoU(Y_s,argMax)  
  entropy = helper.getEntropy(bimap, Y_s)    
  loss = entropy + 1e-5 * helper.regularizer()    
  with tf.variable_scope('bimap'):
    batch = tf.Variable(0)
  LearningRate = 0.001
  DecayRate = 0.99999
  
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
    
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch) 

  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    tf.global_variables_initializer().run()    
    saver_bimap = tf.train.Saver()      
    if isNewTrain: print('Initialized!')
    else :        
        saver_bimap.restore(sess, model.modelName)
        print("Model restored")
    sess.run(tf.local_variables_initializer())  
    feed_dict_valid = {X: valid_data, Y: valid_label, IsTrain :False,Step:0}
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    test_offset = train_data.shape[3] - ensemble  #288 - 12    
    start_offsets = np.arange(test_offset)
    for step in xrange(NUM_EPOCHS):
      model.step = step      
      np.random.shuffle(start_offsets)
      for iter in range((int)(test_offset/1)-1):
          start_offset = start_offsets[iter]
          end_offset = start_offset + ensemble

          #if end_offset < train_data.shape[3]/2:
          #  batch_data_even = train_data[:,:,:,::2][:,:,:,start_offset:end_offset]
          #  sess.run(optimizer, {X: batch_data_even, Y: train_labels, IsTrain:True,Step:step})          
          #  batch_data_odd = train_data[:,:,:,1::2][:,:,:,start_offset:end_offset]                      
          #  sess.run(optimizer, {X: batch_data_odd, Y: train_labels, IsTrain:True,Step:step})
          batch_data = train_data[:,:,:,start_offset:end_offset]

          if AUGMENT<1:              
            feed_dict_flip = {X: np.fliplr(batch_data), Y: np.fliplr(train_labels), IsTrain:True,Step:step}      
            sess.run(optimizer, feed_dict_flip)    
            feed_dict_flip = {X: np.flipud(batch_data), Y: np.flipud(train_labels), IsTrain:True,Step:step}      
            sess.run(optimizer, feed_dict_flip)     

          feed_dict = {X: batch_data[::-1], Y: train_labels[::-1], IsTrain:True,Step:step}      
          _= sess.run(optimizer, feed_dict)          
          feed_dict = {X: batch_data, Y: train_labels, IsTrain:True,Step:step}       
          _,l, iou,lr = sess.run([optimizer,entropy, mean_iou,learning_rate], feed_dict)
          
          if iter % EVAL_FREQUENCY == 0:
            start_time = time.time()
            elapsed_time = time.time() - start_time
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY 
            iou_valid = sess.run(mean_iou, feed_dict_valid)
            iou_test = sess.run(mean_iou, feed_dict_test)
        
            print('%d, %d, %.0fms, L:%.3f, IoU(tr%.0f,va%.0f,te%.0f),lr %.4f, %s' % 
                  (step, iter,takes,l,iou*100,iou_valid*100,iou_test*100,lr*100,now))   
          
            sys.stdout.flush()
            if lr==0 or l>20: 
                print ('lr l has problem  ',lr) 
                return
        
            this_sec = time.time()
            if this_sec - start_sec > 60 * 15 :
                start_sec = this_sec
                save_path = saver_bimap.save(sess, model.modelName)
                now = strftime("%H:%M:%S", localtime())
                print("Model Saved, time:%s" %(now))      
           
    if sess.run(learning_rate)>0: 
        save_path = saver_bimap.save(sess, model.modelName)
        print ('save_path', save_path)      
    
    bimap_mask,mask = sess.run([bimap,argMax], feed_dict= feed_dict_test)            
    DataReader.SaveImage(bimap_mask[:,:,:,1],ImagePath1)

tf.app.run()