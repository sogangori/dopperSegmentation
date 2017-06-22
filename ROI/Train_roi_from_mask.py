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
import Model_roi as model 
folder = model.folder
ImagePath1 = folder+"bimap"
DataReader = DataReader()
EVAL_FREQUENCY = 30
AUGMENT = 1
DATA_SIZE = 12
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 1
isNewTrain = True      

def main(argv=None):        

  ensemble = model.ensemble  
  train_data, train_mask, train_roi, valid_data,valid_mask,valid_roi, test_data,test_mask,test_roi= DataReader.GetROISet(DATA_SIZE,AUGMENT,ensemble);  
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,train_mask.shape[1],train_mask.shape[2]])
  Y = tf.placeholder(tf.float32, [None,2])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  false_co = tf.constant(False)
  
  roi = model.inference(X, IsTrain, Step)   
  loss_y = tf.reduce_mean( tf.square( Y[:,0] - roi[:,0]))
  loss_h = tf.reduce_mean( tf.square( Y[:,1] - roi[:,1]))
  #loss_y = tf.reduce_mean( tf.square( tf.sqrt(Y[:,0]) - tf.sqrt(roi[:,0])))
  #loss_h = tf.reduce_mean( tf.square( tf.sqrt(Y[:,1]) - tf.sqrt(roi[:,1])))
  entropy = loss_y + loss_h
  loss = entropy + 1e-8 * helper.regularizer()    

  diff = tf.reduce_mean(tf.abs(Y - roi),axis=0)
  with tf.variable_scope('roi'):
    batch = tf.Variable(0)
  LearningRate = 0.01
  DecayRate = 0.9999
  
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
    
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch) 

  start_sec = start_time = time.time()
  with tf.Session() as sess:    
    tf.global_variables_initializer().run()    
    saver_bimap = tf.train.Saver()      
    if isNewTrain: print('Initialized!')
    else :        
        saver_bimap.restore(sess, model.modelName)
        print("Model restored")
    sess.run(tf.local_variables_initializer())  
    feed_dict_valid = {X: valid_mask, Y: valid_roi, IsTrain :False,Step:0}
    feed_dict_test = {X: test_mask, Y: test_roi, IsTrain :False,Step:0}    
    test_offset = train_data.shape[3] - ensemble 
    start_offsets = np.arange(test_offset)

    train_mask_ud = np.flipud(train_mask)
    train_roi_ud = train_roi
    train_roi_ud[:,0] = 1-train_roi_ud[:,0]
    train_mask = np.append(train_mask, np.fliplr(train_mask),axis=0)
    train_mask = np.append(train_mask, train_mask_ud,axis=0)
    train_roi = np.append(train_roi, train_roi,axis=0)
    train_roi = np.append(train_roi, train_roi_ud,axis=0)
    for step in xrange(NUM_EPOCHS):
      model.step = step      
      np.random.shuffle(start_offsets)
      for iter in range(test_offset):
          start_offset = start_offsets[iter]
          end_offset = start_offset + ensemble
          
          feed_dict = {X: train_mask, Y: train_roi, IsTrain:True,Step:step}       
          _,ly,lh, lr,roi_train_predic,diff_train = sess.run([optimizer,loss_y,loss_h, learning_rate,roi,diff], feed_dict)          
         
          if iter % EVAL_FREQUENCY == 0:
            start_time = time.time()
            elapsed_time = time.time() - start_time
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY 
            l_val = 1        
            l_val,diff_val = sess.run([entropy,diff], feed_dict= feed_dict_valid)
            print('%d, %d, %.0fms, Ly:%g, Lh:%g,lr %.4f, %s' % 
                  (step, iter,takes,ly,lh, lr*100,now))   
            print ('diff_train/val ', diff_train,diff_val)            
            
            sys.stdout.flush()
            if lr==0 or ly>20: 
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
    
    roi_predict,diff_predict = sess.run([roi,diff], feed_dict= feed_dict_valid)            
    print ('train_roi',train_roi)
    print ('train_predict',roi_train_predic)    
    print ('test_predict',roi_predict)
    print ('diff_train', diff_train)    
    print ('diff_predict', diff_predict)

tf.app.run()