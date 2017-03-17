'''

Operations commonly used in tensorflow

'''

import tensorflow as tf
import numpy as np
import math

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)
         
'''
   enc1 = slim.conv2d(L_image,64,kernel_size=[4,4],stride=2,padding='SAME',
      biases_initializer=None,activation_fn=lrelu,scope='g_enc1',
      weights_initializer=initializer)
'''
def batchnorm(input):
    with tf.variable_scope("batchnorm"):
      # this block looks like it has 3 inputs on the graph unless we do this
      input = tf.identity(input)

      channels = input.get_shape()[3]
      offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
      scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
      mean, variance = tf.nn.moments(input, axes=[0, 1, 2], keep_dims=False)
      variance_epsilon = 1e-5
      normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
      return normalized

def conv2d(batch_input, out_channels, stride):
   with tf.variable_scope("conv"):
      in_channels = batch_input.get_shape()[3]
      filter = tf.get_variable("filter", [4, 4, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
      # [batch, in_height, in_width, in_channels], [filter_width, filter_height, in_channels, out_channels]
      #     => [batch, out_height, out_width, out_channels]
      padded_input = tf.pad(batch_input, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT")
      conv = tf.nn.conv2d(padded_input, filter, [1, stride, stride, 1], padding="VALID")
      return conv

def deconv(batch_input, out_channels):
   with tf.variable_scope("deconv"):
      batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
      filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
      # [batch, in_height, in_width, in_channels], [filter_width, filter_height, out_channels, in_channels]
      #     => [batch, out_height, out_width, out_channels]
      conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * 2, in_width * 2, out_channels], [1, 2, 2, 1], padding="SAME")
      return conv

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

'''
   Sigmoid
'''
def sig(x, name='sig'):
   return tf.nn.sigmoid(x, name)

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
      weight_decay = tf.multiply(tf.nn.l2_loss(var), wd, name='weight_loss')
      weight_decay.set_shape([])
      tf.add_to_collection('losses', weight_decay)
   return var
