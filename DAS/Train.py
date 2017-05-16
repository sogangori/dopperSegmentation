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
from DataReader import DataReader
import Train2loss as helper 
#import TrainSkip3 as helper 
from operator import or_

hiddenImagePath = "./DAS/weights/hidden/"
predictImagePath = "./DAS/weights/predict"
outImagePath = "./DAS/weights/out"
inImagePath = "./DAS/weights/in"

DataReader = DataReader()
EVAL_BATCH_SIZE = 1
EVAL_FREQUENCY = EVAL_BATCH_SIZE
trainH = 64
BATCH_SIZE = 1
NUM_EPOCHS = 100#4000 = 600M
isNewTrain =  True                  


def main(argv=None):        
    
  train_data, train_labels = GetTrainData(isTrain =  True, isRotate = True, count = BATCH_SIZE,trainH= trainH)  
  
  #train_labels = numpy.abs(train_labels)
  train_size = train_data.shape[0]  
  NUM_LABELS = train_labels.shape[1] * train_labels.shape[2]
  print("train_data.shape", train_data.shape)
  print("train_labels.shape", train_labels.shape)
  
  # These placeholder nodes will be fed a batch of training data at each
  # training step using the {feed_dict} argument to the Run() call below.
  train_data_node = tf.placeholder(tf.float32, shape=train_data.shape)
  train_labels_node = tf.placeholder(tf.float32, shape=train_labels.shape)
  eval_data = 0
    
  train_prediction = helper.inference(train_data, True)  
  eval_prediction = 0
        
  mserror = getLoss(train_prediction, train_labels_node)
  
  # Add the regularization term to the loss.
  loss = mserror + 1e-7 * helper.regullarizer()
  
  #tf.scalar_summary("loss", loss)

  # Optimizer: set up a variable that's incremented once per batch and
  # controls the learning rate decay.
  batch = tf.Variable(0)
  LearningRate = 0.1
  DecayRate = 0.99999
  # Decay once per epoch, using an exponential schedule starting at 0.01.
  learning_rate = tf.train.exponential_decay(
      LearningRate,  # Base learning rate.0.01
      batch * BATCH_SIZE,  # Current index into the dataset.
      train_size,  # Decay step.
      DecayRate,  # Decay rate.
      staircase=True)
  
  # Use simple momentum for the optimization.
  optimizer = tf.train.MomentumOptimizer(learning_rate, 0.5).minimize(loss, global_step=batch)
  train_step = tf.train.GradientDescentOptimizer(0.05).minimize(loss) 

  # Small utility function to evaluate a dataset by feeding batches of data to
  # {eval_data} and pulling the results from {eval_predictions}.
  # Saves memory and enables this to run on smaller GPUs.
  def eval_in_batches(data, sess):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
      raise ValueError("batch size for evals larger than dataset: %d" % size)
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

  # Create a local session to run the training.
  start_sec = start_time = time.time()
  config=tf.ConfigProto()
  # config.gpu_options.per_process_gpu_memory_fraction=0.98
  config.gpu_options.allocator_type="BFC"  
  config.log_device_placement=True
  #sess=tf.Session(config=config)
  with tf.Session(config=config) as sess:    

    saver = tf.train.Saver()  
    if isNewTrain:
        tf.initialize_all_variables().run()
        print('Initialized!')
    else :        
        saver.restore(sess, helper.modelName)
        print("Model restored")
        
    #summary_writer = tf.train.SummaryWriter(helper.logName, sess.graph)
    #merged = tf.merge_all_summaries()
    for step in xrange(int(NUM_EPOCHS * train_size) // BATCH_SIZE):
      # Compute the offset of the current minibatch in the data.
      # Note that we could use better randomization across epochs.
      offset = 0
      if train_size - BATCH_SIZE != 0:          
          offset = (step * BATCH_SIZE) % (train_size - BATCH_SIZE)
      batch_data = train_data[offset:(offset + BATCH_SIZE), ...]
      batch_labels = train_labels[offset:(offset + BATCH_SIZE)]
      # This dictionary maps the batch data (as a numpy array) to the
      # node in the graph it should be fed to.
      feed_dict = {train_data_node: batch_data,
                   train_labels_node: batch_labels}
      # Run the graph and fetch some of the nodes.
      _, mse, l, lr, predictions = sess.run(
          [optimizer, mserror, loss, learning_rate, train_prediction], feed_dict=feed_dict)
      #summary_writer.add_summary(summary, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        now = strftime("%H:%M:%S", localtime())
        takes = 1000 * elapsed_time / EVAL_FREQUENCY
        print('Step %d (epoch %.1f), %.0f ms, loss %f, lr %f, time:%s' % 
              (step, float(step) * BATCH_SIZE / train_size,
               takes,l,lr,now))        
        # Add histograms for trainable variables.
        #for var in tf.trainable_variables(): tf.histogram_summary(var.op.name, var)
        # print('Minibatch error: %.1f%%' % error_rate(predictions, batch_labels))
        #print('Validation error: %.1f%%' % error_rate(eval_in_batches(validation_data, sess), validation_labels))
        # print("eval_in_batches(validation_data) size",eval_in_batches(validation_data, sess))
        # print("eval_in_batches(validation_data)",numpy.size( eval_in_batches(validation_data, sess)))
        # print("validation_labels",validation_labels)
        sys.stdout.flush()
        if lr==0 or l>20: 
            print ('lr l has problem  ') 
            return
        
        this_sec = time.time()
        if this_sec - start_sec > 60 * 60 :
            start_sec = this_sec
            save_path = saver.save(sess, helper.modelName)
            now = strftime("%H:%M:%S", localtime())
            print("Model Saved, time:%s" %(now))
        if step % NUM_EPOCHS == 10:
            #saver = tf.train.Saver()
            #save_path = saver.save(sess, helper.modelName)
            #print ('save_path', save_path)
            a=1
        
    if lr>0:
        save_path = saver.save(sess, helper.modelName)

    train_prediction = helper.inference(train_data, False)
    predict = sess.run(train_prediction)  

    SavePredictAsImage(predict,predictImagePath)    
    snr = Accuracy(train_labels, predict)
    
    print ('snr %.3f, mse %.5f, loss %.5f, batch %d, time %s' % (snr, mse, l,BATCH_SIZE,takes))
    print ('save_path', save_path)
      


def getLoss(prediction,labels_node):    
    loss = tf.reduce_mean(tf.square( labels_node - prediction))    
    return loss


def GetTrainData(isTrain = True,isRotate = False,count = 1,trainH = 128):    
    
    #if isRotate and isTrain:  trainingSet,trainingOut = DataReader.GetTrainDataToTensorflowRotateLR(count,LABEL_SIZE_H,LABEL_SIZE_W,LABEL_SIZE_C,isTrain); 
    #else : 
    trainingSet,trainingOut = DataReader.GetTrainDataToTensorflow(count,isTrain, trainH);
        
    return [trainingSet, trainingOut]

def SavePredictAsImage(src, path):    
    DataReader.SaveAsImage(src, path, src.shape[0])
    
def SavePredictAsImageByChannel(src, path):
    DataReader.SaveAsImageByChannel(src, path, src.shape[0])
    
def Accuracy(out ,predict):
    
    return DataReader.SNR(out ,predict)
    

tf.app.run()


