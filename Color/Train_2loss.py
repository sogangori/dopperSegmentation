from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import time

import numpy
from time import localtime, strftime
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf
from operator import or_
from DataReader import DataReader
import Model_2loss as model 

hiddenImagePath = "./Color/weights/hidden/"
predictImagePath = "./Color/weights/predict"
outImagePath = "./Color/weights/out"
inImagePath = "./Color/weights/in"

DataReader = DataReader()
EVAL_BATCH_SIZE = 100
EVAL_FREQUENCY = 5
AUGMENT = 4
DATA_SIZE = 230 #max 230
BATCH_SIZE = DATA_SIZE * 2 # * AUGMENT
NUM_EPOCHS = 30
isNewTrain = not  True                 

def main(argv=None):        
  train_data, train_labels = DataReader.GetDataAug(DATA_SIZE, AUGMENT, isTrain =  True);  
  test_data, test_labels = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  
  print("train_data.shape", train_data.shape)
  print("train_labels.shape", train_labels.shape)
  print("test_data.shape", test_data.shape)
  
  train_size = train_data.shape[0]        
  batch_train_in_shape  = [BATCH_SIZE,train_data.shape[1],train_data.shape[2],train_data.shape[3]]
  batch_train_out_shape = [BATCH_SIZE,train_labels.shape[1],train_labels.shape[2]]
  train_data_node = tf.placeholder(tf.float32, shape=batch_train_in_shape)
  train_labels_node = tf.placeholder(tf.int32, shape=batch_train_out_shape)
  test_data_node = tf.placeholder(tf.float32, shape=test_data.shape)
  test_labels_node = tf.placeholder(tf.int32, shape=test_labels.shape)
    
  train_prediction,train_prediction_small, train_feature_map = model.inference(train_data_node, True)  
  test_prediction,test_prediction_small, test_feature_map = model.inference(test_data, True)  
    
  entropy = getLoss(train_prediction, train_labels_node)
  entropy_small = getLoss_small(train_prediction_small, train_labels_node)
  loss = entropy + entropy_small/2 + 1e-6 * regullarizer()  
  
  accuracy = tf.reduce_mean(tf.cast(tf.equal(train_prediction[:,:,:,1],tf.cast(train_labels_node, tf.float32)), dtype= tf.float32))    
  test_accuracy = tf.reduce_mean(tf.cast(tf.equal(test_prediction[:,:,:,1],test_labels), dtype= tf.float32))
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
    predictions = numpy.ndarray(shape=(size, NUM_LABELS), dtype=numpy.float32)
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
        tf.initialize_all_variables().run()
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
    
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: train_data[offset:(offset + BATCH_SIZE)], 
                   train_labels_node: train_labels[offset:(offset + BATCH_SIZE)]}
      # Run the graph and fetch some of the nodes.
      _, l,acc,acc_s, lr, predictions = sess.run(
          [optimizer, entropy,entropy_small, accuracy, learning_rate, train_prediction], feed_dict=feed_dict)
      #summary_writer.add_summary(summary, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        now = strftime("%H:%M:%S", localtime())
        takes = 1000 * elapsed_time / EVAL_FREQUENCY
        print('%d, epo %.1f, %.0f ms, entropy %f, %f, acc %.3f, lr %f, %s' % 
              (step, float(step) * BATCH_SIZE / train_size,takes,l,acc,acc_s, lr,now))        

        feed_dict_test = {test_data_node: test_data, test_labels_node: test_labels}
        test_acc = sess.run([test_accuracy], feed_dict=feed_dict_test)
         
        # Add histograms for trainable variables.
        #for var in tf.trainable_variables(): tf.histogram_summary(var.op.name, var)
        # print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        #print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
        # print("eval_in_batches(validation_data) size",eval_in_batches(validation_data, sess))
        # print("eval_in_batches(validation_data)",numpy.size( eval_in_batches(validation_data, sess)))
        # print("validation_labels",validation_labels)
        sys.stdout.flush()
        if lr==0 or l>20: 
            print ('lr l has problem  ',lr) 
            #return
        
        this_sec = time.time()
        if this_sec - start_sec > 60 * 30 :
            start_sec = this_sec
            save_path = saver.save(sess, model.modelName)
            now = strftime("%H:%M:%S", localtime())
            print("Model Saved, time:%s" %(now))
        if step % NUM_EPOCHS == 10:
            #saver = tf.train.Saver()
            #save_path = saver.save(sess, model.modelName)
            #print ('save_path', save_path)
            a=1
        
    if lr>0: 
        save_path = saver.save(sess, model.modelName)
        print ('save_path', save_path)      
            
    predict_test,predict_test_small,test_feature_map = sess.run(model.inference(test_data))
    test_acc = sess.run(test_accuracy, feed_dict=feed_dict_test)
    print('accuracy train:%.2f, test:%.2f ' % (acc, test_acc))            
    
    predict_test = numpy.array(predict_test)
    print('predict_test' , predict_test.shape)         
    DataReader.SaveAsImage(predict_test[:,:,:,1], predictImagePath, EVAL_BATCH_SIZE, maxCount = 10)
    #DataReader.SaveFeatureMap(test_feature_map, "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE, maxCount = 1)

def getLoss_small(prediction,labels_node): 
    predict_shape = prediction.get_shape().as_list()   
    prediction =  tf.reshape(prediction, [-1, model.LABEL_SIZE_C])
    shape = labels_node.get_shape().as_list()
    print("*********** predict_shape",predict_shape)
    labels_node = tf.reshape(labels_node, [shape[0],shape[1],shape[2],1])
    labels_node = tf.image.resize_bilinear(labels_node, [predict_shape[2], predict_shape[1]])
    label_reshape = tf.reshape(labels_node, [-1])
    label_reshape = tf.cast(label_reshape , tf.int32)
    print ('prediction',prediction)
    print ('labels_node',labels_node)
    print ('labels_node_reshape',label_reshape)
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = label_reshape)
    return  tf.reduce_mean(entropy)    

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
    return  tf.reduce_mean(entropy)    

def regullarizer():
    variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)    
    regula=0
    for i in range(len(variables)):
        regula +=  tf.nn.l2_loss(variables[i])
    return regula

tf.app.run()


