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
EVAL_FREQUENCY = 10
AUGMENT = 3
DATA_SIZE = 12 
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 1
isNewTrain = not True      

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
  train_data, train_labels,valid_data,valid_label, test_data,test_label = DataReader.GetData3(DATA_SIZE,AUGMENT,ensemble);  
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
  final_bimap = tf.cast( tf.add(bimap_fore_idx,trimap_fore_idx)-trimap_back_idx>0 , tf.float32)
  mean_iou = getIoU(Y,argMax)
  mean_iou_final = getIoU(Y,tf.cast(final_bimap,tf.int32))
  entropy = getLossMSE(bimap, Y,trimap)    
  loss = entropy + 1e-5 * regularizer()    
  with tf.variable_scope('bimap'):
    batch = tf.Variable(0)
  LearningRate = 0.01
  DecayRate = 0.999
  
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
  
  variable_bimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'bimap')
  variable_trimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'trimap')
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch, var_list=variable_bimap) 

  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    tf.global_variables_initializer().run()
    saver_trimap = tf.train.Saver(variable_trimap)  
    saver_bimap = tf.train.Saver(variable_bimap)  
    saver_trimap.restore(sess, modelTrimap.modelName)
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

          if end_offset < train_data.shape[3]/2:
            batch_data_even = train_data[:,:,:,::2][:,:,:,start_offset:end_offset]
            sess.run(optimizer, {X: batch_data_even, Y: train_labels, IsTrain:True,Step:step})          
            batch_data_odd = train_data[:,:,:,1::2][:,:,:,start_offset:end_offset]                      
            sess.run(optimizer, {X: batch_data_odd, Y: train_labels, IsTrain:True,Step:step})
          
          batch_data = train_data[:,:,:,start_offset:end_offset]
          feed_dict = {X: batch_data[::-1], Y: train_labels[::-1], IsTrain:True,Step:step}      
          _,l, iou_r,iou_final_r,lr = sess.run([optimizer,entropy, mean_iou,mean_iou_final, learning_rate], feed_dict)          
          feed_dict = {X: batch_data, Y: train_labels, IsTrain:True,Step:step}       
          _,l, iou,iou_final,lr = sess.run([optimizer,entropy, mean_iou,mean_iou_final, learning_rate], feed_dict)
          
          if iter % EVAL_FREQUENCY == 0:
            start_time = time.time()
            elapsed_time = time.time() - start_time
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY 
            iou_valid = sess.run(mean_iou, feed_dict_valid)#여러개          
            iou_test = sess.run(mean_iou, feed_dict_test)
        
            print('e%d,i%d,%.0fms,L:%.3f,IoU_r(%.0f,%.0f), IoU(%.0f,tr%.0f,va%.0f,te%.0f),lr %.4f, %s' % 
                  (step, iter,takes,l,iou_r*100,iou_final_r*100,iou*100,iou_valid*100,iou_test*100,iou_final*100,lr*100,now))   
          
            sys.stdout.flush()
            if lr==0 or l>20: 
                print ('lr l has problem  ',lr) 
                return
        
            this_sec = time.time()
            if this_sec - start_sec > 60 * 20 :
                start_sec = this_sec
                save_path = saver_bimap.save(sess, model.modelName)
                now = strftime("%H:%M:%S", localtime())
                print("Model Saved, time:%s" %(now))      
           
    if sess.run(learning_rate)>0: 
        save_path = saver_bimap.save(sess, model.modelName)
        print ('save_path', save_path)      
    
    bimap_mask,final_bimap_mask = sess.run([bimap,final_bimap], feed_dict= feed_dict_test)        
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

def regularizer():
    regula=0
    for var in tf.trainable_variables():         
        regula +=  tf.nn.l2_loss(var)
    return regula

tf.app.run()