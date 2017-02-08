'''

Operations commonly used in tensorflow

'''


import tensorflow as tf
import numpy as np
import math


'''
   Activation summary for tensorboard.
'''
def activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

'''
   Places a variable on the GPU
'''
def _variable_on_gpu(name, shape, initializer):
   with tf.device('/gpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
   return var


'''
   Creates a variable with weight decay
'''
def _variable_with_weight_decay(name, shape, stddev, wd):
   var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
   if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
      weight_decay.set_shape([])
      tf.add_to_collection('losses', weight_decay)
   return var




'''
   taken from: http://stackoverflow.com/a/33950177 and
   https://github.com/carpedm20/DCGAN-tensorflow/blob/master/ops.py

   SO, look here https://github.com/carpedm20/DCGAN-tensorflow/issues/59
   looks like exponential moving average cannot be used under reuse=True in
   the new tensorflow, so having some issues. Going to copy paste links to help

   https://github.com/carpedm20/DCGAN-tensorflow/issues/57
   https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/models/batch_norm.py#L64


'''
def batch_norm(x, name, epsilon=1e-5, momentum=0.9, train=True):
   
   shape = x.get_shape().as_list()
   ema   = tf.train.ExponentialMovingAverage(decay=momentum)

   if train is not None:
      with tf.variable_scope(name) as scope:
         beta  = tf.get_variable('beta', [shape[-1]], initializer=tf.constant_initializer(0.))
         gamma = tf.get_variable('gamma', [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))

         try:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
         except:
            batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')

         # THIS FIXED IT - JUST A HACK THOUGH
         with tf.variable_scope(tf.get_variable_scope(), reuse=False):
            ema_apply_op = ema.apply([batch_mean, batch_var])
         ema_mean, ema_var = ema.average(batch_mean), ema.average(batch_var)

         with tf.control_dependencies([ema_apply_op]):
            mean, var = tf.identity(batch_mean), tf.identity(batch_var)
   else:
      mean, var = ema_mean, ema_var   
   
   # normed
   return tf.nn.batch_norm_with_global_normalization(x, mean, var, beta, gamma, epsilon, scale_after_normalization=True) 


'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)


'''
   Regular relu
'''
def relu(x, name='relu'):
   return tf.nn.relu(x, name)


'''
   Tanh
'''
def tanh(x, name='tanh'):
   return tf.nn.tanh(x, name)

def sig(x, name='sig'):
   return tf.nn.sigmoid(x, name)

'''

   A fully connected layer.

   Inputs
   x: A tensor. It can be an image, previous fully connected layer, conv layer, etc
   hidden_units: An int. Specifies how many hidden layers the layer will have.
   flatten: Bool. Whether or not to flatten the input. If the input is an image for example, this is True
   name: String. A name for the layer for saving and reloading models. Must be unique

   Returns
   A tensor containing the output of the fully connected layer. No activation.

'''
def fc_layer(x, hidden_units, flatten, name):
   with tf.variable_scope(name) as scope:
      input_shape = x.get_shape().as_list()
      if flatten:
         dim = input_shape[1]*input_shape[2]*input_shape[3]
         x_processed = tf.reshape(x, [-1,dim])
      else:
         dim = input_shape[1]
         x_processed = x

      weights = _variable_with_weight_decay('weights', shape=[dim,hidden_units],stddev=0.01, wd=0.0005)
      biases = _variable_on_gpu('biases', [hidden_units], tf.constant_initializer(0.01))

      return tf.add(tf.matmul(x_processed,weights), biases, name=name)

'''
   Transpose convolution (deconv)
'''
def conv2d_transpose(x, kernel_size, stride, num_channels, name):
   with tf.variable_scope(name) as scope:
      input_channels = x.get_shape()[3]

      weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_channels,input_channels], stddev=0.1, wd=0.0005)
      biases = _variable_on_gpu('biases',[num_channels],tf.constant_initializer(0.1))
      batch_size = tf.shape(x)[0]
      output_shape = tf.pack([tf.shape(x)[0], tf.shape(x)[1]*stride, tf.shape(x)[2]*stride, num_channels])
      conv = tf.nn.conv2d_transpose(x, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      return conv_biased


'''
   Convolution

   Inputs
   x: A [MxNxDxB] tensor, where B is the batch_size.
   kernel_size: An int. The size of the kernel
   stride: An int. The stride number
   num_features: An int. The number of feature maps (channels).
   name: String. Must be unique

   Returns
   The convolved image or whatever it was. No activation function.
'''
def conv2d(x, kernel_size, stride, num_features, name):
   with tf.variable_scope(name) as scope:
      input_channels = x.get_shape()[3]
      weights = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_channels, num_features], stddev=0.1, wd=0.0005)
      biases = _variable_on_gpu('biases', [num_features], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(x, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      return conv_biased

