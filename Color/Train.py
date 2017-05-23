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
import Model_bn_narrow as model 
#http://angusg.com/writing/2016/12/28/optimizing-iou-semantic-segmentation.html
hiddenImagePath = "./Color/weights/hidden/"
predictImagePath = "./Color/weights/predict"
outImagePath = "./Color/weights/out"
inImagePath = "./Color/weights/in"

DataReader = DataReader()
EVAL_BATCH_SIZE = 10
EVAL_FREQUENCY = 5
AUGMENT = 3
DATA_SIZE = 100#360+131+130 #max 360 + 131 + 130 = 621
BATCH_SIZE = np.int(DATA_SIZE / 2)  # * AUGMENT
NUM_EPOCHS = 5
isNewTrain = not True                 
#82 43%, bn: 83 39
def main(argv=None):        
  train_data, train_labels,train_help = DataReader.GetDataAug(DATA_SIZE, AUGMENT, isTrain =  True);  
  test_data, test_labels,test_help = DataReader.GetDataAug(EVAL_BATCH_SIZE,1, isTrain =  False);  
  
  print("train_data.shape", train_data.shape)
  print("train_labels.shape", train_labels.shape)
  print("test_data.shape", test_data.shape)
  
  train_size = train_data.shape[0]          
  X = tf.placeholder(tf.float32, [None,train_data.shape[1],train_data.shape[2],train_data.shape[3]])
  Y = tf.placeholder(tf.int32, [None,train_labels.shape[1],train_labels.shape[2]])
  IsTrain = tf.placeholder(tf.bool)
  Step = tf.placeholder(tf.int32)
  
  prediction, feature_map = model.inference(X, IsTrain, Step)    
  argMax = tf.cast( tf.arg_max(prediction,3), tf.int32)
  accuracy = tf.contrib.metrics.accuracy(argMax,Y)     
  mean_iou = getIoU(Y,argMax)
  entropy = getLoss(prediction, Y)
  loss_iou = 1 - mean_iou   
  loss = entropy + 1e-5 * regullarizer()    
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
      feed_dict = {X: train_data[offset:(offset + BATCH_SIZE)],
                   Y: train_labels[offset:(offset + BATCH_SIZE)],
                   IsTrain:True,Step:step}      
      _, l,l2,acc, iou,lr = sess.run([optimizer, entropy,loss_iou, accuracy,mean_iou, learning_rate], feed_dict)
      #summary_writer.add_summary(summary, step)
      if step % EVAL_FREQUENCY == 0:
        elapsed_time = time.time() - start_time
        start_time = time.time()
        now = strftime("%H:%M:%S", localtime())
        takes = 1000 * elapsed_time / EVAL_FREQUENCY
        feed_dict_test = {X: test_data, Y: test_labels,IsTrain:False,Step:step}
        iou_test = sess.run(mean_iou, feed_dict_test)
        
        print('%d/%.1f, %.0f ms, loss(%.3f,%.3f),IoU(%g,%.3f),lr %.4f, %s' % 
              (step, float(step) * BATCH_SIZE / train_size,takes,l,l2,iou,iou_test, lr*100,now))   
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
        
    if lr>0: 
        save_path = saver.save(sess, model.modelName)
        print ('save_path', save_path)      
            
    predict,feature_map_,test_acc = sess.run([prediction, feature_map,accuracy], feed_dict=feed_dict_test)
    print('accuracy train:%.2f, test:%.2f' % (iou, iou_test))                    
    DataReader.SaveAsImage(predict[:,:,:,1], predictImagePath, EVAL_BATCH_SIZE, maxCount = 10)
    #DataReader.SaveFeatureMap(feature_map, "./Color/weights/featureMap/fm", EVAL_BATCH_SIZE, maxCount = 1)

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

def regullarizer():
    regula=0
    for var in tf.trainable_variables(): 
        regula +=  tf.nn.l2_loss(var)
    return regula

tf.app.run()


