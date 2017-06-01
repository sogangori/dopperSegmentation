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
import Model_narrow as model 
#http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
hiddenImagePath = "./DAS_ /weights/hidden/"
predictImagePath = "./DAS_COLOR/weights/predict"
outImagePath = "./DAS_COLOR/weights/out"
inImagePath = "./DAS_COLOR/weights/in"

DataReader = DataReader()
EVAL_FREQUENCY = 20
AUGMENT = 1
DATA_SIZE = 12
EVAL_BATCH_SIZE = DATA_SIZE 
BATCH_SIZE = np.int(DATA_SIZE)  # * AUGMENT 
NUM_EPOCHS = 1
test_ratio = 0.2
isNewTrain = not True      

def main(argv=None):        

  train_data, train_labels = DataReader.GetData(DATA_SIZE);  
    
  print("train_data.shape", train_data.shape)
  print("train_labels.shape", train_labels.shape)
    
  train_size = train_data.shape[0]          
  test_offset = np.int(train_data.shape[3]* (1.0 - test_ratio))
  test_count = train_size - test_offset
  print ('test_offset',test_offset,'test_count',test_count)
  ensemble = model.ensemble  
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],ensemble])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  
  prediction = model.inference(X, IsTrain, Step)    
  argMax = tf.cast( tf.arg_max(prediction,3), tf.int32)
  accuracy = tf.contrib.metrics.accuracy(argMax,Y)     
  mean_iou = getIoU(Y,argMax)
  entropy = getLoss(prediction, Y)  
  loss = entropy + 1e-5 * regularizer()    
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
      model.step = step
      
      for iter in range(test_offset):
          
          ensemble_start = iter % (test_offset-ensemble+1) 
          batch_data = train_data[:,:,:,ensemble_start:ensemble_start + ensemble]          
          feed_dict = {X: batch_data[offset:(offset + BATCH_SIZE)],
                       Y: train_labels[offset:(offset + BATCH_SIZE)],
                       IsTrain:True,Step:step}      
          _, l,acc, iou,lr = sess.run([optimizer, entropy, accuracy,mean_iou, learning_rate], feed_dict)
          #summary_writer.add_summary(summary, step)
          if iter % EVAL_FREQUENCY == 0:
            elapsed_time = time.time() - start_time
            start_time = time.time()
            now = strftime("%H:%M:%S", localtime())
            takes = 1000 * elapsed_time / EVAL_FREQUENCY
            #batch_data = train_data[:,:,:,test_offset:test_offset+iter% test_count]
            batch_data = train_data[:,:,:,test_offset:test_offset+ensemble]
            feed_dict_test = {X: batch_data, Y: train_labels, IsTrain :False,Step:0}
            iou_test = sess.run(mean_iou, feed_dict_test)
        
            print('%d/%.1f, %.0f ms, loss %.3f,IoU(%g,%.3f),lr %.4f, %s' % 
                  (step, iter,takes,l,iou,iou_test, lr*100,now))   
            # Add histograms for trainable variables.
            #for var in tf.trainable_variables(): tf.histogram_summary(var.op.name, var)
    
            sys.stdout.flush()
            if lr==0 or l>20: 
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
            
    batch_data = train_data[:,:,:,200:203]
    predict,test_acc = sess.run([prediction, accuracy], feed_dict= {X: batch_data, Y: train_labels, IsTrain :False,Step:0})    
    DataReader.SaveAsImage(predict[:,:,:,1], predictImagePath, EVAL_BATCH_SIZE)
    

def getIoU(a,b):
    a = tf.round(a)    
    b = tf.round(b)
    trn_labels=tf.reshape(a, [-1])
    logits=tf.reshape(b, [-1])
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels))
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)))
    iou = inter/union
    return tf.cast(iou,tf.float32)

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

def regularizer():
    regula=0
    for var in tf.trainable_variables(): 
        regula +=  tf.nn.l2_loss(var)
    return regula

tf.app.run()


