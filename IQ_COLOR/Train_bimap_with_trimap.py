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
predictImagePath = "./IQ_COLOR/weights/predict"
ImagePath2 = "./IQ_COLOR/weights/final"

DataReader = DataReader()
EVAL_BATCH_SIZE = 10
EVAL_FREQUENCY = 5
AUGMENT = 4
DATA_SIZE = 250#360+131+130 #max 360 + 131 + 130 = 621
BATCH_SIZE = np.int(DATA_SIZE/2)  # * AUGMENT
NUM_EPOCHS = 50
isNewTrain = not True        

def main(argv=None):
  train_data, train_labels = DataReader.GetDataAug(DATA_SIZE, AUGMENT, isTrain =  True);  
  test_data, test_labels = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  
  print("train_data.shape", train_data.shape)
  print("train_labels.shape", train_labels.shape)
  print("test_data.shape", test_data.shape)
  
  train_size = train_data.shape[0]          
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],train_data.shape[3]])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
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
  final_bimap = tf.cast( tf.add(bimap_fore_idx,trimap_fore_idx)-trimap_back_idx>0 , tf.float32)
  mean_iou = helper.getIoU(Y,argMax)
  mean_iou_final = helper.getIoU(Y,tf.cast(final_bimap,tf.int32))
  entropy = helper.getLossMSE(bimap, Y,trimap)    
  loss = entropy + 1e-5 * helper.regularizer()    
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
  optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=batch,var_list=variable_bimap)
 
  start_sec = start_time = time.time()
  with tf.Session() as sess:    
    tf.global_variables_initializer().run()
    saver_trimap = tf.train.Saver(variable_trimap)  
    saver_bimap = tf.train.Saver(variable_bimap)  
    saver_trimap.restore(sess, modelTrimap.modelName)
    if isNewTrain: 
        print('Initialized!')
    else :        
        saver_bimap.restore(sess, model.modelName)
        print("Model restored")
    sess.run(tf.local_variables_initializer())  
    #summary_writer = tf.train.SummaryWriter(model.logName, sess.graph)
    #merged = tf.merge_all_summaries()

    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = 0
      if train_size - BATCH_SIZE != 0:          
          offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    
      model.step = step
      feed_dict = {X: train_data[offset:(offset + BATCH_SIZE)],
                   Y: train_labels[offset:(offset + BATCH_SIZE)],
                   IsTrain:True,Step:step}      
      _, l,l2, iou,iou_final,lr = sess.run([optimizer, entropy,loss, mean_iou, mean_iou_final,learning_rate], feed_dict)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        now = strftime("%H:%M:%S", localtime())
        takes = 1000 * elapsed_time / EVAL_FREQUENCY
        feed_dict_test = {X: test_data, Y: test_labels,IsTrain:False,Step:0}
        iou_test,iou_test_final = sess.run([mean_iou,mean_iou_final], feed_dict_test)
        
        print('%d/%.1f, %.0f ms, loss(%.3f,%.3f),IoU(%.2f,%.2f), IoU_final: %.3f, %.3f),lr %.4f, %s' % 
              (step, float(step) * BATCH_SIZE / train_size,takes,l,l2,iou,iou_test,iou_final,iou_test_final, lr*100,now))   
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
        
    if lr>0: 
        save_path = saver_bimap.save(sess, model.modelName)
        print ('save_path', save_path)      
            
    predict= sess.run(final_bimap, feed_dict=feed_dict_test)                    
    DataReader.SaveAsImage(predict, ImagePath2, EVAL_BATCH_SIZE)
    #DataReader.SaveFeatureMap(feature_map, "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE)

tf.app.run()