import tensorflow as tf
import numpy as np
import sys
sys.path.insert(0, '../../ops/')
from tf_ops import lrelu, conv2d, batch_norm, fc_layer, conv2d_transpose


'''
   Discriminator D

   Inputs:

   Outputs:

'''
def discriminator(image, batch_size, reuse=False):
   if reuse:
      tf.get_variable_scope().reuse_variables()

   conv1 = lrelu(conv2d(image, 5, 2, 64, 'd_conv1'))
   conv2 = lrelu(batch_norm(conv2d(conv1, 5, 2, 128, 'd_conv2'), 'd_bn1'))
   conv3 = lrelu(batch_norm(conv2d(conv2, 5, 2, 256, 'd_conv3'), 'd_bn2'))
   conv4 = lrelu(batch_norm(conv2d(conv3, 5, 2, 512, 'd_conv4'), 'd_bn3'))
   conv4 = fc_layer(conv4, 1, True, 'd_conv4_lin')
   #conv4 = fc_layer(tf.reshape(conv4, [batch_size, -1]), 1, True, 'd_conv4_lin')
   return tf.nn.sigmoid(conv4), conv4


'''
   Generator G

   Inputs:

   Outputs:

'''
def generator(z, batch_size):

   # project and reshape z
   z = fc_layer(z, 1024*4*4, False, 'g_lin')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])

   conv_t1 = tf.nn.relu((batch_norm(conv2d_transpose(z, 5, 2, 256, 'g_conv_t1'), 'g_bn1')))
   conv_t2 = tf.nn.relu((batch_norm(conv2d_transpose(conv_t1, 5, 2, 128, 'g_conv_t2'), 'g_bn2')))
   conv_t3 = tf.nn.relu((batch_norm(conv2d_transpose(conv_t2, 5, 2, 1, 'g_conv_t3'), 'g_bn3')))

   return tf.nn.tanh(conv_t3)


'''

   Loss function for D and G

   Inputs:

   Outputs:

'''
def loss(logits, labels):
   return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits, labels))

