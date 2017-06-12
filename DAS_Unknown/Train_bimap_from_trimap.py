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
import Model_bimap_from_trimap as model 
import Model_trimap_2x as modelTrimap 
folder = "./DAS_Unknown/weights/"
hiddenImagePath = folder+"hidden/"
ImagePath1 = folder+"bimap"
ImagePath2 = folder+"final"
outImagePath = folder+"out"
inImagePath = folder+"in"

DataReader = DataReader()
EVAL_FREQUENCY = 20
AUGMENT = 1
DATA_SIZE = 12 
BATCH_SIZE = np.int(DATA_SIZE)  
NUM_EPOCHS = 2
isNewTrain = not True      

def getLossMSE(bimap, labels_node):        
    shape = tf.shape(bimap)
    label = tf.one_hot(labels_node,2)    
    error = tf.square(label - bimap)     
    return tf.reduce_mean(error)

def main(argv=None):        

  ensemble = modelTrimap.ensemble  
  train_data, train_labels,test_data,test_label = DataReader.GetDataTrainTest(DATA_SIZE,AUGMENT,ensemble);  
  train_size = train_data.shape[0]           
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  false_co = tf.constant(False)
  trimap = modelTrimap.inference(X, false_co, Step)
  bimap = model.inference(trimap, IsTrain, Step)
  argMax = tf.cast( tf.arg_max(bimap,3), tf.int32)    
  mean_iou = getIoU(Y,argMax)  
  entropy = getLossMSE(bimap, Y)    
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
    if isNewTrain:
        print('Initialized!')
    else :        
        saver_bimap.restore(sess, model.modelName)
        print("Model restored")
    sess.run(tf.local_variables_initializer())  
    feed_dict_test = {X: test_data, Y: test_label, IsTrain :False,Step:0}
    test_offset = train_data.shape[3] - ensemble  #288 - 12    
    start_offsets = np.arange(test_offset)
    for step in xrange(NUM_EPOCHS):
      model.step = step      
      np.random.shuffle(start_offsets)
      for iter in range((int)(test_offset/1)):
          batch_data = train_data[:,:,:,start_offsets[iter]:start_offsets[iter] + ensemble]          
          feed_dict = {X: batch_data, Y: train_labels, IsTrain:True,Step:step}      
          _,l, iou,lr = sess.run([optimizer,entropy, mean_iou,learning_rate], feed_dict)
          
          if iter % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY           
            iou_test = sess.run(mean_iou, feed_dict_test)
        
            print('e%d,i%d,%.0fms,L:%.3f,IoU(%.2f,%.2f),lr %.4f, %s' % 
                  (step, iter,takes,l,iou*100,iou_test*100,lr*100,now))   
          
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
    
    bimap_mask = sess.run(bimap, feed_dict= feed_dict_test)        
    print ('bimap_mask',bimap_mask.shape)
    DataReader.SaveImage(bimap_mask[:,:,:,1],ImagePath1)
    
    

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