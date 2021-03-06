﻿from __future__ import absolute_import
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
import Model_autoEncoder as model 

hiddenImagePath = "./unsupervised/weights/hidden/"
predictImagePath = "./unsupervised/weights/predict"
outImagePath = "./unsupervised/weights/out"
inImagePath = "./unsupervised/weights/in"

DataReader = DataReader()
EVAL_BATCH_SIZE = 10
EVAL_FREQUENCY = 5
AUGMENT = 2
DATA_SIZE = 200#360+131+130 #max 360 + 131 + 130 = 621
BATCH_SIZE = np.int(DATA_SIZE / 1)  # * AUGMENT
NUM_EPOCHS = 20
isNewTrain = True                 

def main(argv=None):        
  train_data, train_labels,train_help = DataReader.GetDataAug(DATA_SIZE, AUGMENT, isTrain =  True);  
  test_data, test_labels,test_help = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  
  print("train_data.shape", train_data.shape)
  print("test_data.shape", test_data.shape)
  
  train_size = train_data.shape[0]        
  batch_train_in_shape  = [BATCH_SIZE,train_data.shape[1],train_data.shape[2],train_data.shape[3]]  
  train_data_node = tf.placeholder(tf.float32, shape=batch_train_in_shape)  
  test_data_node = tf.placeholder(tf.float32, shape=test_data.shape)
    
  train_prediction, train_feature_map = model.inference(train_data_node, True)  
  test_prediction, test_feature_map = model.inference(test_data_node, True)  
  entropy = getLoss(train_data_node,train_prediction)
  loss = entropy + 1e-5 * regullarizer()  
    
  accuracy = loss    
  test_accuracy = getLoss(test_data,test_prediction)
  
  #tf.scalar_summary("loss", loss)

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
  optimizer = tf.train.AdamOptimizer(learning_rate, 0.5).minimize(loss, global_step=batch)
 
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
    NUM_LABELS = train_labels.shape[1] * train_labels.shape[2]
    predictions = np.ndarray(shape=(size, NUM_LABELS), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
      end = begin + EVAL_BATCH_SIZE
      if end <= size:
        predictions[begin:end, :] = sess.run(
            eval_prediction, feed_dict={eval_data: data[begin:end, ...]})
      else:
        batch_predictions = sess.run(
            eval_prediction,feed_dict={eval_data: data[-EVAL_BATCH_SIZE:, ...]})
        predictions[begin:, :] = batch_predictions[begin - size:, :]
    return predictions

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
        
    #summary_writer = tf.train.SummaryWriter(model.logName, sess.graph)
    #merged = tf.merge_all_summaries()
    
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = 0
      if train_size - BATCH_SIZE != 0:          
          offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
    
      # This dictionary maps the batch data (as a np array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: train_data[offset:(offset + BATCH_SIZE)]}
      # Run the graph and fetch some of the nodes.
      _, l,acc, lr, predictions = sess.run(
          [optimizer, entropy, accuracy, learning_rate, train_prediction], feed_dict=feed_dict)
      #summary_writer.add_summary(summary, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        now = strftime("%H:%M:%S", localtime())
        takes = 1000 * elapsed_time / EVAL_FREQUENCY
        feed_dict_test = {test_data_node: test_data}
        test_acc = sess.run(test_accuracy, feed_dict=feed_dict_test)
        print('%d/%.1f, %.0f ms, loss %f, acc %.1f, %.1f, lr %.4f, %s' % 
              (step, float(step) * BATCH_SIZE / train_size,takes,l,acc*100, test_acc*100,lr*100,now))        
                 
        # Add histograms for trainable variables.
        #for var in tf.trainable_variables(): tf.histogram_summary(var.op.name, var)
    
        sys.stdout.flush()
        if lr==0 or l>20: 
            print ('lr l has problem  ',lr) 
            return
        if (not isNewTrain) and l>1.5:
            print ('lr l has problem 2 ',lr) 
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
            
    predict_test,test_feature_map = sess.run(model.inference(test_data))
    test_acc = sess.run(test_accuracy, feed_dict=feed_dict_test)
    print('accuracy train:%.2f, test:%.2f' % (acc, test_acc))            
    pred = np.array(np.argmin(predict_test,3), int)
    print('pred ' , pred .shape)    
    predict_test = np.array(predict_test)
    print('predict_test' , predict_test.shape)         
    DataReader.SaveAsImage(predict_test[:,:,:,1], predictImagePath, EVAL_BATCH_SIZE, maxCount = 10)
    DataReader.SaveFeatureMap(test_feature_map, "./unsupervised/weights/featureMap/fm", EVAL_BATCH_SIZE, maxCount = 1)


def getLoss(prediction,labels_node):    
    return tf.reduce_mean(tf.square(prediction-labels_node))

def regullarizer():
    regula=0
    for var in tf.trainable_variables(): 
        regula +=  tf.nn.l2_loss(var)
    return regula

def Gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(shape = input_layer.get_shape(), mean = 0.0, stddev = std, dtype = tf.float32) 
    return input_layer + noise

tf.app.run()


