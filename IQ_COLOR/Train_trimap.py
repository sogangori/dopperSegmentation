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

DataReader = DataReader()
EVAL_BATCH_SIZE = 10
EVAL_FREQUENCY = 5
AUGMENT = 3
DATA_SIZE = 250
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 50
isNewTrain = not True

def main(argv=None):        

  train_data, train_label = DataReader.GetDataAug(DATA_SIZE, AUGMENT, isTrain =  True);  
  test_data, test_label = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  print ('test_data',test_data.shape)
  print ('train_label',train_label.shape)
  train_size = train_label.shape[0]
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],train_data.shape[3]])
  Y = tf.placeholder(tf.int32, [None,train_label.shape[1],train_label.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  
  trimap,featureMap = model.inference(X, IsTrain, Step)
  argMax = tf.cast( tf.arg_max(trimap,3), tf.int32)  
  known = tf.cast( argMax < 2, tf.float32)  
  unknown = tf.cast( argMax > 1, tf.float32)  
  background = tf.cast( argMax < 1, tf.float32)
  unknown_mean = tf.reduce_mean(unknown)  
  background_mean = tf.reduce_mean(background)  
  foreground_mean = 1.0 - unknown_mean - background_mean
  
  Y_known = tf.multiply(tf.cast(Y, tf.float32), known)
  argMax_known = tf.multiply(tf.cast(argMax, tf.float32), known)
  mean_iou_known = helper.getIoU(Y_known,argMax_known)
  mean_iou = helper.getIoU(Y,argMax)
  entropy = helper.getLossMSE_penalty(trimap, Y)    
  trimap_prob = tf.nn.softmax(trimap)
  trimap_prob_2d = tf.reshape(trimap_prob, [-1,3])
  loss_unknown = tf.reduce_mean(tf.square(trimap_prob_2d[:,2])/2)
  loss = entropy + 5e-2 * loss_unknown + 1e-5 * helper.regularizer()    

  batch = tf.Variable(0)
  LearningRate = 0.01
  DecayRate = 0.999
  
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
  
  # Use simple momentum for the optimization.
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
    sess.run(tf.local_variables_initializer())  
    
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = 0
      if train_size - BATCH_SIZE != 0:          
          offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph it should be fed to.
      model.step = step
      feed_dict = {X: train_data[offset:(offset + BATCH_SIZE)],
                   Y: train_label[offset:(offset + BATCH_SIZE)],
                   IsTrain:True,Step:step}            
      sess.run(optimizer, feed_dict= {X: feed_dict[X][::-1],Y: feed_dict[Y][::-1], IsTrain:True,Step:step})
      _,unkno,fore,back, entro,l_unknown, iou,iou_known,lr = sess.run(
              [optimizer,unknown_mean, foreground_mean,background_mean,entropy,loss_unknown, mean_iou, mean_iou_known,learning_rate], feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        now = strftime("%H:%M:%S", localtime())
        takes = 1000 * elapsed_time / EVAL_FREQUENCY
        feed_dict_test = {X: test_data, Y: test_label,IsTrain:False,Step:0}
        iou_test = sess.run(mean_iou, feed_dict_test)
        
        iter = float(step) * BATCH_SIZE / train_size,takes,
        print('%d,%.0fms,tri(%.0f,%.0f,%.0f),L:(%.3f,%.4f),IoU(train:%.0f,test:%.0f),Iou_k:%.1f,lr %.4f' % 
                  (step, takes,back*100,fore*100,unkno*100,entro,l_unknown,iou*100,iou_test*100, iou_known*100,lr*100))   
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
        
    if lr>0: 
        save_path = saver.save(sess, model.modelName)
        print ('save_path', save_path)          
    
    trimap_mask,unknown_mask = sess.run([trimap,unknown], feed_dict= feed_dict_test)    
    DataReader.SaveAsImage(unknown_mask, ImagePath2, trimap_mask.shape[0])    
    print ('trimap_mask',trimap_mask.shape)
    DataReader.SaveImage(trimap_mask,ImagePath1)

tf.app.run()