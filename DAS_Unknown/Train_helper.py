import tensorflow as tf

def GetMaskWithBimapTrimap(bimap, trimap):
    bimap_fore_idx = tf.cast(tf.arg_max(bimap,3), tf.float32)    
    trimap_argMax = tf.arg_max(trimap,3)
    trimap_unknown_idx = tf.cast(trimap_argMax > 1, tf.float32)
    trimap_back_idx = tf.cast(trimap_argMax < 1, tf.float32)    
    trimap_fore_idx = tf.ones_like(trimap_back_idx) - trimap_back_idx - trimap_unknown_idx   
    final_mask = tf.cast( tf.add(bimap_fore_idx,trimap_fore_idx)-trimap_back_idx>0 , tf.int32)
    return final_mask

def getEntropy(prediction,labels_node):    
    shape = prediction.get_shape().as_list()
    prediction =  tf.reshape(prediction, [-1, shape[3]])
    label_reshape = tf.reshape(labels_node, [-1])
    print ('prediction',prediction)
    print ('labels_node',labels_node)
    print ('labels_node_reshape',label_reshape)
    #and labels of shape [batch_size]. But higher dimensions are supported.
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = prediction, labels = label_reshape)
    return  tf.reduce_mean(entropy)  


def getLossMSE_penalty(trimap, labels_node):    
    shape = tf.shape(trimap)
    label = tf.one_hot(labels_node,2)
    fore_weight = tf.cast(labels_node, tf.float32)*2.0
    bimap = trimap[:,:,:,0:2]
    error = tf.square(label - bimap)
    trimap_prob = tf.nn.softmax(trimap)
    known_idx = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)
    known_prob = known_idx * (1-trimap_prob[:,:,:,2])+fore_weight
    known_re = tf.reshape(known_prob, [-1,shape[1],shape[2],1])    
    weight = tf.concat([known_re,known_re],3)      
    error_bimap = tf.multiply(error,weight)
    return tf.reduce_mean(error_bimap)

def getLossMSE_focus_unknown(bimap, labels_node, trimap):   
     
    shape = tf.shape(bimap)
    label = tf.one_hot(labels_node,2)    
    error = tf.square(label - bimap)
    bimap_prob = tf.nn.softmax(bimap)
    unknown_idx = tf.cast(tf.arg_max(trimap,3) > 1, tf.float32) + 0.1
    unknown_4d = tf.reshape(unknown_idx, [-1,shape[1],shape[2],1])    
    unknown_mask = tf.concat([unknown_4d,unknown_4d],3)    
    error_bimap = tf.multiply(error,unknown_mask)    
    return tf.reduce_mean(error_bimap)

def getLossMSE(bimap, labels_node):        
    shape = tf.shape(bimap)
    label = tf.one_hot(labels_node,2)    
    error = tf.square(label - bimap)     
    return tf.reduce_mean(error)

def getLoss_log(trimap, labels_node):    
    shape = tf.shape(trimap)
    label = tf.one_hot(labels_node,2)
    trimap_prob = tf.nn.softmax(trimap)
    bimap = tf.nn.softmax(trimap[:,:,:,0:2])
    #cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) *tf.log(1 - hypothesis))
    error = -1*(tf.multiply(label, tf.log(bimap) ) + tf.multiply((1-label), tf.log(1-bimap)))    
    known_idx = tf.cast( tf.arg_max(trimap,3) < 2, tf.float32)
    known_prob = known_idx * (1-trimap_prob[:,:,:,2])
    known_re = tf.reshape(known_prob, [-1,shape[1],shape[2],1])    
    weight = tf.concat([known_re,known_re],3)    
    error_bimap = tf.multiply(error,weight)    
    return tf.reduce_mean(error_bimap)

def getLoss_log_class(trimap, label, channel):   
    n = tf.shape(trimap)[0]     
    shape = tf.shape(trimap)
    Y = tf.reshape(label, [-1])
    n = tf.shape(trimap)[0]
    trimap_prob = tf.nn.softmax(trimap)
    foregroundMask = trimap_prob[:,:,:,channel]
    foregroundMask = tf.reshape(foregroundMask, [-1])
    Y_f = tf.cast(Y,tf.float32)
    cost = Y_f * tf.log(foregroundMask)
    return -tf.reduce_mean(cost)    

def getLoss_entropy_class(trimap, label, channel):   
    n = tf.shape(trimap)[0]     
    shape = tf.shape(trimap)
    Y = tf.reshape(label, [-1])
    n = tf.shape(trimap)[0]
    bimap = trimap[:,:,:,0:2]
    bimap = tf.reshape(bimap, [-1,2])
    entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits = bimap, labels = Y)    
    cost = tf.reduce_mean(entropy)    
    return cost

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

def getIoULoss(a,b):
    n = tf.shape(a)[0]
    a = tf.cast(a, tf.float32)
    b = tf.cast(b, tf.float32)
    trn_labels=tf.reshape(a, [n,-1])
    logits=tf.reshape(b, [n,-1])
    inter = tf.multiply(logits,trn_labels)
    print ('inter',inter)
    union= tf.subtract(tf.add(logits,trn_labels),inter)
    
    loss = -tf.log(tf.divide(inter,union+0.1))
    loss = tf.reduce_mean(loss)
    return loss

def returnZero():
    return 0

def regularizer():
    regula=0    
    for var in tf.trainable_variables():        
        regula += tf.cond(tf.rank(var) > 2, lambda: tf.nn.l2_loss(var), lambda: tf.constant(0.0))
    return regula

def GetPruningIndex(src,std_cut):
    src_shape = tf.shape(src)
    src = tf.reshape(src,[-1])
    batch_mean, batch_var = tf.nn.moments(x = src,axes=[0])
    batch_std = tf.sqrt(batch_var)
    #print ('mean, var', p(batch_mean),p( batch_var),p( tf.sqrt(batch_var)))
    cut0 = batch_mean - batch_std * std_cut
    cut1 = batch_mean + batch_std * std_cut
    #print ('cut', p(cut0), p(cut1))
    prun_index0 = tf.cast(src<cut0, tf.float32)
    prun_index1 = tf.cast(src>cut1, tf.float32)
    prun_index =  tf.add(prun_index0, prun_index1)
    return tf.reshape(prun_index,src_shape)

def Pruning(variables,  pruning_masks):
    k=0
    
    for var,pruning_mask in zip(variables,pruning_masks): 
        k+=1
        var = tf.multiply(var, pruning_mask)
    return k
