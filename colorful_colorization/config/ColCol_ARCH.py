import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np
import math
import sys

#Leaky RELU : https://arxiv.org/pdf/1502.01852.pdf
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)



def batch_norm(x):
    with tf.variable_scope('batchnorm'):
      x = tf.identity(x)

      channels = x.get_shape()[3]
      offset = tf.get_variable('offset', [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
      scale = tf.get_variable('scale', [channels], dtype=tf.float32, initializer=tf.random_normal_initializer(1.0, 0.02))
      mean, variance = tf.nn.moments(x, axes=[0, 1, 2], keep_dims=False)
      variance_epsilon = 1e-5
      normalized = tf.nn.batch_normalization(x, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
      return normalized

def conv2d(x, out_channels, stride=2,kernel_size=4):
   with tf.variable_scope('conv2d'):
      in_channels = x.get_shape()[3]
      kernel = tf.get_variable('kernel', [kernel_size, kernel_size, in_channels, out_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
      # [batch, in_height, in_width, in_channels], [kernel_width, kernel_height, in_channels, out_channels]
      #     => [batch, out_height, out_width, out_channels]
      padded_input = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='CONSTANT')
      conv = tf.nn.conv2d(padded_input, kernel, [1, stride, stride, 1], padding='VALID')
      return conv

def conv2d_transpose(x, out_channels, stride=2, kernel_size=4):
   with tf.variable_scope('conv2d_transpose'):
      batch, in_height, in_width, in_channels = [int(d) for d in x.get_shape()]
      kernel = tf.get_variable('kernel', [kernel_size, kernel_size, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
      conv = tf.nn.conv2d_transpose(x, kernel, [batch, in_height * 2, in_width * 2, out_channels], [1, stride, stride, 1], padding='SAME')
      return conv


def netG(L_image, batch_size, num_gpu):
   if   num_gpu == 0: Machines = ['/cpu:0']
   elif num_gpu == 1: Machines = ['/gpu:0']
   elif num_gpu == 2: Machines = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: Machines = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: Machines = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   for d in Machines:
      with tf.device(d):
	  with tf.variable_scope('conv1_1'):
	        conv1_1 = conv2d(L_image, 64, kernel_size=3, stride=1)
	        conv1_1 = tf.nn.relu(conv1_1)
	  with tf.variable_scope('conv1_2'):
	        conv1_2 = conv2d(conv1_1, 64, kernel_size=3, stride=2)
	        conv1_2 = tf.nn.relu(conv1_2)
	        conv1_2 = batch_norm(conv1_2)
	  with tf.variable_scope('conv2_1'):
	        conv2_1 = conv2d(conv1_2, 128, kernel_size=3, stride=1)
	        conv2_1 = tf.nn.relu(conv2_1)
	  with tf.variable_scope('conv2_2'):
	        conv2_2 = conv2d(conv2_1, 128, kernel_size=3, stride=2)
	        conv2_2 = tf.nn.relu(conv2_2)
	        conv2_2 = batch_norm(conv2_2)
	  with tf.variable_scope('conv3_1'):
	        conv3_1 = conv2d(conv2_2, 256, kernel_size=3, stride=1)
	        conv3_1 = tf.nn.relu(conv3_1)
	  with tf.variable_scope('conv3_2'):
	        conv3_2 = conv2d(conv3_1, 256, kernel_size=3, stride=1)
	        conv3_2 = tf.nn.relu(conv3_2)
	  with tf.variable_scope('conv3_3'):
	        conv3_3 = conv2d(conv3_2, 256, kernel_size=3, stride=2)
	        conv3_3 = tf.nn.relu(conv3_3)
	        conv3_3 = batch_norm(conv3_3)      
	  with tf.variable_scope('conv4_1'):
	        conv4_1 = conv2d(conv3_3, 512, kernel_size=3, stride=1)
	        conv4_1 = tf.nn.relu(conv4_1)
	  with tf.variable_scope('conv4_2'):
	        conv4_2 = conv2d(conv4_1, 512, kernel_size=3, stride=1)
	        conv4_2 = tf.nn.relu(conv4_2)
	  with tf.variable_scope('conv4_3'):
	        conv4_3 = conv2d(conv4_2, 512, kernel_size=3, stride=1)
	        conv4_3 = tf.nn.relu(conv4_3)
	        conv4_3 = batch_norm(conv4_3)   
	  with tf.variable_scope('conv5_1'):
	        conv5_1 = conv2d(conv4_3, 512, kernel_size=3, stride=1)
	        conv5_1 = tf.nn.relu(conv5_1)
	  with tf.variable_scope('conv5_2'):
	        conv5_2 = conv2d(conv5_1, 512, kernel_size=3, stride=1)
	        conv5_2 = tf.nn.relu(conv5_2)
	  with tf.variable_scope('conv5_3'):
	        conv5_3 = conv2d(conv5_2, 512, kernel_size=3, stride=1)
	        conv5_3 = tf.nn.relu(conv5_3)
	        conv5_3 = batch_norm(conv5_3)
	  with tf.variable_scope('conv6_1'):
	        conv6_1 = conv2d(conv5_3, 512, kernel_size=3, stride=1)
	        conv6_1 = tf.nn.relu(conv6_1)
	  with tf.variable_scope('conv6_2'):
	        conv6_2 = conv2d(conv6_1, 512, kernel_size=3, stride=1)
	        conv6_2 = tf.nn.relu(conv6_2)
	  with tf.variable_scope('conv6_3'):
	        conv6_3 = conv2d(conv6_2, 512, kernel_size=3, stride=1)
	        conv6_3 = tf.nn.relu(conv6_3)
	        conv6_3 = batch_norm(conv6_3)
	  with tf.variable_scope('conv7_1'):
	        conv7_1 = conv2d(conv6_3, 256, kernel_size=3, stride=1)
	        conv7_1 = tf.nn.relu(conv7_1)
	  with tf.variable_scope('conv7_2'):
	        conv7_2 = conv2d(conv7_1, 256, kernel_size=3, stride=1)
	        conv7_2 = tf.nn.relu(conv7_2)  
	  with tf.variable_scope('conv7_3'):
	        conv7_3 = conv2d(conv6_2, 256, kernel_size=3, stride=1)
	        conv7_3 = tf.nn.relu(conv7_3)
	        conv7_3 = batch_norm(conv7_3)
	  with tf.variable_scope('conv8_1'):
	        conv8_1 = conv2d(conv7_3, 128, kernel_size=3, stride=1)
	        conv8_1 = tf.nn.relu(conv8_1)
	  with tf.variable_scope('conv8_2'):
	        conv8_2 = conv2d(conv8_1, 128, kernel_size=3, stride=1)
	        conv8_2 = tf.nn.relu(conv8_2)  
	  with tf.variable_scope('conv8_3'):
	        conv8_3 = conv2d(conv8_2, 128, kernel_size=3, stride=1)
	        conv8_3 = tf.nn.relu(conv8_3)
	        conv8_3 = batch_norm(conv8_3)
	  
	  
	  with tf.variable_scope('conv9_1'):
	        conv9_1 = conv2d_transpose(conv8_3, 64, stride=2, kernel_size=4)
	        conv9_1 = tf.nn.relu(conv9_1) 
	  with tf.variable_scope('conv9_2'):
	        conv9_2 = conv2d_transpose(conv9_1, 32, stride=2, kernel_size=4)
	        conv9_2 = tf.nn.relu(conv9_2)
	  with tf.variable_scope('conv9_3'):
	        conv9_3 = conv2d_transpose(conv9_2, 2, stride=2, kernel_size=4)
	        conv9_3 = tf.nn.relu(conv9_3)
	        conv9_3 = batch_norm(conv9_3)
	        
	        
   print '< GEN >'
   print 'conv1_1:',conv1_1
   print 'conv1_2:',conv1_2
   print 'conv2_1:',conv2_1
   print 'conv2_2:',conv2_2
   print 'conv3_1:',conv3_1
   print 'conv3_2:',conv3_2
   print 'conv3_3:',conv3_3
   print 'conv4_1:',conv4_1
   print 'conv4_2:',conv4_2
   print 'conv4_3:',conv4_3
   print 'conv5_1:',conv5_1
   print 'conv5_2:',conv5_2
   print 'conv5_3:',conv5_3
   print 'conv6_1:',conv6_1
   print 'conv6_2:',conv6_2
   print 'conv6_3:',conv6_3
   print 'conv7_1:',conv7_1
   print 'conv7_2:',conv7_2
   print 'conv7_3:',conv7_3
   print 'conv8_1:',conv8_1
   print 'conv8_2:',conv8_2
   print 'conv8_3:',conv8_3
   print
   print 'conv9_1:',conv9_1
   print 'conv9_2:',conv9_2
   print 'conv9_3:',conv9_3
   
   
   print '< END GEN >'
   print

   tf.add_to_collection('vars', conv1_1)
   tf.add_to_collection('vars', conv1_2)
   tf.add_to_collection('vars', conv2_1)
   tf.add_to_collection('vars', conv2_2)
   tf.add_to_collection('vars', conv3_1)
   tf.add_to_collection('vars', conv3_2)
   tf.add_to_collection('vars', conv3_3)
   tf.add_to_collection('vars', conv4_1)
   tf.add_to_collection('vars', conv4_2)
   tf.add_to_collection('vars', conv4_3)
   tf.add_to_collection('vars', conv5_1)
   tf.add_to_collection('vars', conv5_2)
   tf.add_to_collection('vars', conv5_3)
   tf.add_to_collection('vars', conv6_1)
   tf.add_to_collection('vars', conv6_2)
   tf.add_to_collection('vars', conv6_3)
   tf.add_to_collection('vars', conv7_1)
   tf.add_to_collection('vars', conv7_2)
   tf.add_to_collection('vars', conv7_3)
   tf.add_to_collection('vars', conv8_1)
   tf.add_to_collection('vars', conv8_2)
   tf.add_to_collection('vars', conv8_3)
   tf.add_to_collection('vars', conv9_1)
   tf.add_to_collection('vars', conv9_2)
   tf.add_to_collection('vars', conv9_3)

   
   return conv9_3

