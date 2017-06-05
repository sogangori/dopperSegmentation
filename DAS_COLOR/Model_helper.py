import tensorflow as tf
import time

keep_prop = 1.0
isDrop = False


def dropout(src):
    src = tf.cond(isDrop, lambda:tf.nn.dropout(src, keep_prop), lambda:identity(src))
    return src

def identity(src):
    return src

def resize(src, dstH, dstW,interpol = 1):
    
    if interpol == 0: return tf.image.resize_nearest_neighbor(src, [dstH, dstW])
    elif interpol == 1: return tf.image.resize_bilinear(src, [dstH, dstW])
    else: return tf.image.resize_bicubic(src, [dstH, dstW])

def conv2d(src, weights, bias):
    src = dropout(src)
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    return tf.nn.bias_add(conv,bias)


def conv2dRelu(src, weights, bias):
    conv = conv2d(src,weights,bias)
    return tf.nn.relu(conv) 

def conv2dReluStride2(src, weights, bias):
    src = dropout(src)
    conv = tf.nn.conv2d(src,weights,strides=[1, 2, 2, 1],padding='SAME')
    conv = tf.nn.bias_add(conv,bias)
    return tf.nn.relu(conv) 

def Gaussian_noise_layer(input_layer, std):
    noise = tf.random_normal(tf.shape(input_layer), mean = 0.0, stddev = std, dtype = tf.float32) 
    return input_layer + noise

def Gaussian_noise_volumn_layer(input_layer, std):
    input_shape = tf.shape(input_layer)
    shape = [input_shape[0],1,1,input_shape[3]]
    noise = tf.random_normal(shape , mean = 1.0, stddev = std, dtype = tf.float32) 
    return input_layer * noise

def Gaussian_noise_Add(input_layer, stdAlpha, stdBeta):
    src = Gaussian_noise_volumn_layer(input_layer,stdAlpha)
    return Gaussian_noise_layer(src,stdBeta)   

def upConv(src, weights, bias,up_shape):
    output = tf.nn.conv2d_transpose(src,weights,output_shape=up_shape, strides=[1, 2, 2, 1],padding='SAME')
    output = tf.nn.bias_add(output,bias)  
    return output

def upConvRelu(src, weights, bias,up_shape):
    output = tf.nn.conv2d_transpose(src,weights,output_shape=up_shape, strides=[1, 2, 2, 1],padding='SAME')
    output = tf.nn.bias_add(output,bias)  
    return tf.nn.relu(output) 

def depthwiseConv2dRelu(src, weights, bias):
    src = dropout(src)
    output = tf.nn.depthwise_conv2d(src,weights,strides=[1,1,1,1],padding='SAME')
    output = tf.nn.bias_add(output, bias)
    output = tf.nn.relu(output)
    return output 

def batchNormal(src, beta,gamma):
    batch_mean, batch_var = tf.nn.moments(x=src,axes=[0,1,2])
    out = tf.nn.batch_normalization(x=src, mean=batch_mean, variance=batch_var,offset=beta,scale=gamma,variance_epsilon=1e-3)    
    return tf.nn.relu(out) 

def batch_norm(src, beta, gamma, isTrain):
    batch_mean, batch_var = tf.nn.moments(x=src,axes=[0,1,2])
    ema = tf.train.ExponentialMovingAverage(decay=0.5)

    def mean_var_with_update():
        ema_apply_op = ema.apply([batch_mean, batch_var])

        with tf.control_dependencies([ema_apply_op]):

            return tf.identity(batch_mean), tf.identity(batch_var)
    mean, var = tf.cond(isTrain, mean_var_with_update, lambda:(ema.average(batch_mean),ema.average(batch_var)))    
    return tf.nn.batch_normalization(src, mean,var, beta, gamma,1e-3)

def conv2dBN(src, weights, beta,gamma, isTrain):
    src = dropout(src)
    conv = tf.nn.conv2d(src,weights,strides=[1, 1, 1, 1],padding='SAME')
    return batchNormal(conv,beta,gamma)

def conv2dBN_Relu(src, weights, beta,gamma, isTrain):
    src_s = dropout(src)
    conv = tf.nn.conv2d(src_s,weights,strides=[1, 1, 1, 1],padding='SAME')
    bn = batchNormal(conv,beta,gamma)
    return tf.nn.relu(bn)

def avg_pool_resize(src, step):
    if step%3==1: src= tf.nn.avg_pool(src,pool_stride2,strides=pool_stride2,padding='SAME')
    elif step%3==2:src= tf.nn.avg_pool(src,pool_stride3,strides=pool_stride3,padding='SAME')
    return src