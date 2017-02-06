import tensorflow as tf
import numpy as np
import math

def activation_summary(x):
  tensor_name = x.op.name
  tf.histogram_summary(tensor_name + '/activations', x)
  tf.scalar_summary(tensor_name + '/sparsity', tf.nn.zero_fraction(x))

'''

'''
def _variable_on_gpu(name, shape, initializer):
   with tf.device('/gpu:0'):
      var = tf.get_variable(name, shape, initializer=initializer)
   return var


'''

'''
def _variable_with_weight_decay(name, shape, stddev, wd):
   var = _variable_on_gpu(name, shape, tf.truncated_normal_initializer(stddev=stddev))
   if wd:
      weight_decay = tf.mul(tf.nn.l2_loss(var), wd, name='weight_loss')
      weight_decay.set_shape([])
      tf.add_to_collection('losses', weight_decay)
   return var


def batch_norm(x, name, epsilon=1e-5, momentum=0.9, train=True):
   shape = x.get_shape().as_list()
   ema   = tf.train.ExponentialMovingAverage(decay=momentum)

   if train:
      with tf.variable_scope(name) as scope:
         beta  = tf.get_variable('beta', [shape[-1]], initializer=tf.constant_initializer(0.))
         gamma = tf.get_variable("gamma", [shape[-1]], initializer=tf.random_normal_initializer(1., 0.02))

         try:
            batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
         except:
            batch_mean, batch_var = tf.nn.moments(x, [0,1], name='moments')

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
'''
def lrelu(x, leak=0.1, name='lrelu'):
   return tf.maximum(leak*x, x)


def relu(x, name='relu'):
   return tf.nn.relu(x)

'''
'''
def fc_layer(inputs, hidden_units, flatten, name):
   with tf.variable_scope(name) as scope:
      input_shape = inputs.get_shape().as_list()
      if flatten:
         dim = input_shape[1]*input_shape[2]*input_shape[3]
         inputs_processed = tf.reshape(inputs, [-1,dim])
      else:
         dim = input_shape[1]
         inputs_processed = inputs

      weights = _variable_with_weight_decay('weights', shape=[dim,hidden_units],stddev=0.01, wd=0.0005)
      biases = _variable_on_gpu('biases', [hidden_units], tf.constant_initializer(0.01))

      return tf.add(tf.matmul(inputs_processed,weights), biases, name=name)

'''
   Transpose convolution (deconv)
'''
def conv2d_transpose(inputs, kernel_size, stride, num_channels, name):
   with tf.variable_scope(name) as scope:
      input_channels = inputs.get_shape()[3]

      weights = _variable_with_weight_decay('weights', shape=[kernel_size,kernel_size,num_channels,input_channels], stddev=0.1, wd=0.0005)
      biases = _variable_on_gpu('biases',[num_channels],tf.constant_initializer(0.1))
      batch_size = tf.shape(inputs)[0]
      output_shape = tf.pack([tf.shape(inputs)[0], tf.shape(inputs)[1]*stride, tf.shape(inputs)[2]*stride, num_channels])
      conv = tf.nn.conv2d_transpose(inputs, weights, output_shape, strides=[1,stride,stride,1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      return conv_biased


'''
   Convolution
'''
def conv2d(inputs, kernel_size, stride, num_features, name):
   with tf.variable_scope(name) as scope:
      input_channels = inputs.get_shape()[3]
      weights = _variable_with_weight_decay('weights', shape=[kernel_size, kernel_size, input_channels, num_features], stddev=0.1, wd=0.0005)
      biases = _variable_on_gpu('biases', [num_features], tf.constant_initializer(0.1))
      conv = tf.nn.conv2d(inputs, weights, strides=[1, stride, stride, 1], padding='SAME')
      conv_biased = tf.nn.bias_add(conv, biases)
      return conv_biased

