import tensorflow as tf

def getLossMSE_penalty(trimap, labels_node):    
    shape = tf.shape(trimap)
    label = tf.one_hot(labels_node,2)
    bimap = trimap[:,:,:,0:2]
    error = tf.square(label - bimap)
    trimap_prob = tf.nn.softmax(trimap)
    known_idx = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)
    known_prob = known_idx * (1-trimap_prob[:,:,:,2])
    known_re = tf.reshape(known_prob, [-1,shape[1],shape[2],1])    
    weight = tf.concat([known_re,known_re],3)    
    error_bimap = tf.multiply(error,weight)    
    return tf.reduce_mean(error_bimap)

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

def getIoU(a,b):
    n = tf.shape(a)[0]
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    trn_labels=tf.reshape(a, [n,-1])
    logits=tf.reshape(b, [n,-1])
    inter=tf.reduce_sum(tf.multiply(logits,trn_labels),axis=1)
    print ('inter',inter)
    union=tf.reduce_sum(tf.subtract(tf.add(logits,trn_labels),tf.multiply(logits,trn_labels)),axis=1)    
    iou = tf.reduce_mean(tf.divide(inter,union))
    return iou

def regularizer():
    regula=0    
    for var in tf.trainable_variables():         
        regula +=  tf.nn.l2_loss(var)
    return regula

