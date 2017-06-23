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
import Model_trimap_normal_s as model 

folder = "./DAS_Unknown/weights/"
hiddenImagePath = folder+"hidden/"
ImagePath1 = folder+"trimap"
ImagePath2 = folder+"unknown"
ImagePath3 = folder+"fore"

DataReader = DataReader()
EVAL_FREQUENCY = 10
AUGMENT = 1
DATA_SIZE = 24
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 1
isNewTrain = not True      
isSparseTrain =not  False

def main(argv=None):        

  ensemble = model.ensemble  
  train_data, train_labels,valid_data,valid_label, test_data,test_label = DataReader.GetData3(DATA_SIZE,AUGMENT,ensemble);  
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  pruning_masks = []
  trimap = model.inference(X, IsTrain, Step)        
  argMax = tf.cast( tf.arg_max(trimap,3), tf.int32)  
  known = tf.cast( argMax < 2, tf.float32)  
  unknown = tf.cast( argMax > 1, tf.float32)  
  background = tf.cast( argMax < 1, tf.float32)
  foreground = tf.ones_like(argMax, tf.float32) - background -  unknown
  unknown_mean = tf.reduce_mean(unknown)  
  background_mean = tf.reduce_mean(background)  
  foreground_mean = tf.reduce_mean(foreground)  
    
  Y_known = tf.multiply(tf.cast(Y, tf.float32), known)
  argMax_known = tf.multiply(tf.cast(argMax, tf.float32), known)
  mean_iou_known = helper.getIoU(Y_known,argMax_known)
  mean_iou = helper.getIoU(Y,argMax)
  entropy = helper.getLossMSE_penalty(trimap, Y)    
  trimap_prob = tf.nn.softmax(trimap)  
  loss_unknown = tf.reduce_mean(tf.square(trimap[:,:,:,2])/2)
  regular =  1e-5 * helper.regularizer()    
  loss = entropy + 1e-2 * loss_unknown + regular

  batch = tf.Variable(0)
  LearningRate = 0.01
  DecayRate = 0.9999
  
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
  
  # Use simple momentum for the optimization.
  variable_trimap = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope = 'trimap')
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch) 
          
  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction=0.98
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=False
  with tf.Session(config=config) as sess:    
    saver = tf.train.Saver()  
    if isNewTrain:
        tf.global_variables_initializer().run()
        print('Initialized!')
    else :        
        saver.restore(sess, model.modelName)
        print("Model restored")
        if isSparseTrain:
          std_cut = 0.5
          for var in variable_trimap:        
              pruning_mask = tf.cond(tf.rank(var) > 2, lambda:  tf.ones_like(var, tf.float32), lambda: helper.GetPruningIndex(var,std_cut))
              pruning_masks.append(pruning_mask)
        
    sess.run(tf.local_variables_initializer())  
    #summary_writer = tf.train.SummaryWriter(model.logName, sess.graph)
    #merged = tf.merge_all_summaries()
    feed_dict_valid = {X: valid_data, Y: valid_label, IsTrain :False,Step:0}
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    test_offset = train_data.shape[3] - ensemble 
    start_offsets = np.arange(test_offset)
    for step in xrange(NUM_EPOCHS):
      model.step = step      
      np.random.shuffle(start_offsets)
      for iter in range((int)(test_offset)):
          batch_data = train_data[:,:,:,start_offsets[iter]:start_offsets[iter] + ensemble]
          feed_dict_reverse = {X: batch_data[::-1], Y: train_labels[::-1], IsTrain:True,Step:step}
          _,unkno = sess.run([optimizer,unknown_mean], feed_dict_reverse)                
          sparseK = helper.Pruning(variable_trimap,pruning_masks)
          feed_dict = {X: batch_data, Y: train_labels, IsTrain:True,Step:step}      
          _,unkno,fore,back, entro,l_unknown, iou,iou_known,lr = sess.run(
              [optimizer, unknown_mean, foreground_mean,background_mean,entropy,regular, mean_iou, mean_iou_known,learning_rate], feed_dict)
          sparseK = helper.Pruning(variable_trimap,pruning_masks)
          if AUGMENT<2:                            
              feed_dict_flip = {X: np.fliplr(batch_data), Y: np.fliplr(train_labels), IsTrain:True,Step:step}      
              _,iou_valid,iou_valid_known = sess.run([optimizer,mean_iou,mean_iou_known], feed_dict_flip)    
              sparseK = helper.Pruning(variable_trimap,pruning_masks)
              feed_dict_flip = {X: np.flipud(batch_data), Y: np.flipud(train_labels), IsTrain:True,Step:step}      
              _,iou_valid,iou_valid_known = sess.run([optimizer,mean_iou,mean_iou_known], feed_dict_flip)              
              sparseK = helper.Pruning(variable_trimap,pruning_masks)
                
          if iter % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY
            iou_valid,iou_valid_known = sess.run([mean_iou,mean_iou_known], feed_dict_valid)
                    
            print('%.2f,%d, tri(%.0f,%.0f,%.0f),L:(%.5f,%.4f),IoU(tr%.0f,k%.0f,va%.1f,k%.1f), lr %.4f' % 
                  (step+ 1.0*iter/test_offset,sparseK, back*100,fore*100,unkno*100,entro,l_unknown,
                   iou*100,iou_known*100,iou_valid*100,iou_valid_known*100,lr*100))   
                 
            sys.stdout.flush()
            if lr==0 or entro>20: 
                print ('lr l has problem  ',lr) 
                return
        
            this_sec = time.time()
            if this_sec - start_sec > 60 * 20 :
                start_sec = this_sec
                save_path = saver.save(sess, model.modelName)
                now = strftime("%H:%M:%S", localtime())
                print("Model Saved, time:%s" %(now))      
           
    if sess.run(learning_rate)>0: 
        save_path = saver.save(sess, model.modelName)
        print ('save_path', save_path)      
    
    iou_valid,iou_valid_known = sess.run([mean_iou,mean_iou_known], feed_dict_valid)
    trimap_mask,foreground_mask, unknown_mask,iou_test,iou_test_known = sess.run([trimap,foreground, unknown,mean_iou,mean_iou_known], feed_dict= feed_dict_test)
    print('IoU (tr%.3f, k%.3f, va%.3f, va_k%.3f, te%.3f, te_k%.3f)' % (iou, iou_known,iou_valid,iou_valid_known,iou_test,iou_test_known))   
                     
    DataReader.SaveAsImage(unknown_mask, ImagePath2, trimap_mask.shape[0])    
    DataReader.SaveAsImage(foreground_mask, ImagePath3, trimap_mask.shape[0])        
    DataReader.SaveImage(trimap_mask,ImagePath1)

tf.app.run()