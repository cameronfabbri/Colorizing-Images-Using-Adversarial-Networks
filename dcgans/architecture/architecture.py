'''

   The artitecture for DCGANs. Things to note:

   Generator:
      - Input is put through a matrix multiplication to a 4-D tensor
      - Use ReLU for all layers except the output, which uses TanH -> actually using LeakyRelu

   Discriminator:
      - Do not apply batch norm to the input layer
      - The last convolution layer is flattened and fed into a sigmoid output
      - Use Leaky Relu in all layers

   Both:
      - Use strided convolutions instead of pooling layers
      - Use batch norm (except where stated above)

   Provide noise in the form of dropout on several layers of the generator on test AND train time.
   
   Trying out Tensorflow SLIM, check here for functions used like conv2d, fully connected, etc
   https://github.com/tensorflow/tensorflow/blob/master/tensorflow/contrib/layers/python/layers/layers.py

'''

import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, '../../ops/')
from tf_ops import fc_layer, conv2d, conv2d_transpose, lrelu, tanh, sig, batch_norm
from tf_ops import lrelu, relu

'''
def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
   shape = input_.get_shape().as_list()
   with tf.variable_scope(scope or "Linear"):
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
      bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))       
      return tf.matmul(input_, matrix) + bias
'''

def generator(gray_image, batch_size, dataset, train=True):

   conv1 = slim.convolution2d(gray_image, 32, 3, stride=1, scope='g_conv1', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv1 = lrelu(conv1)
   print 'g_conv1:',conv1
  
   conv2 = slim.convolution2d(conv1, 32, 3, stride=1, scope='g_conv2', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv2 = lrelu(conv2)
   print 'g_conv2:',conv2

   conv3 = slim.convolution2d(conv2, 64, 3, stride=1, scope='g_conv3', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv3 = lrelu(conv3)
   print 'g_conv3:',conv3

   conv4 = slim.convolution2d(conv3, 64, 3, stride=1, scope='g_conv4', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv4 = lrelu(conv4)
   print 'g_conv4:',conv4
  
   conv5 = slim.convolution2d(conv4, 128, 3, stride=1, scope='g_conv5', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv5 = lrelu(conv5)
   print 'g_conv5:',conv5

   conv6 = slim.convolution2d(conv5, 128, 3, stride=1, scope='g_conv6', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv6 = lrelu(conv6)
   print 'g_conv6:',conv6

   conv7 = slim.convolution2d(conv6, 256, 3, stride=1, scope='g_conv7', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv7 = lrelu(conv7)
   print 'g_conv7:',conv7

   #conv8 = slim.convolution2d(conv7, 512, 3, stride=1, scope='g_conv8', activation_fn=None, normalizer_fn=slim.batch_norm)
   #conv8 = lrelu(conv8)
   #print 'g_conv8:',conv8
   conv8 = conv7

   conv9 = slim.convolution2d(conv8, 256, 3, stride=1, scope='g_conv9', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv9 = lrelu(conv9)
   print 'g_conv9:',conv9
  
   conv10 = slim.convolution2d(conv9, 128, 3, stride=1, scope='g_conv10', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv10 = lrelu(conv10)
   print 'g_conv5:',conv10

   conv11 = slim.convolution2d(conv10, 64, 3, stride=1, scope='g_conv11', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv11 = lrelu(conv11)
   print 'g_conv11:',conv11

   conv12 = slim.convolution2d(conv11, 32, 1, stride=1, scope='g_conv12', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv12 = lrelu(conv12)
   print 'g_conv12:',conv12
   
   conv13 = slim.convolution2d(conv12, 32, 1, stride=1, scope='g_conv13', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv13 = lrelu(conv13)
   print 'g_conv13:',conv13
   
   conv14 = slim.convolution2d(conv13, 16, 1, stride=1, scope='g_conv14', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv14 = lrelu(conv14)
   print 'g_conv14:',conv14
   
   conv15 = slim.convolution2d(conv14, 8, 1, stride=1, scope='g_conv15', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv15 = lrelu(conv15)
   print 'g_conv15:',conv15
   
   conv16 = slim.convolution2d(conv15, 3, 3, stride=1, scope='g_conv16', activation_fn=None, normalizer_fn=slim.batch_norm)
   print 'g_conv16:',conv16

   #return tf.nn.tanh(conv16)
   return lrelu(conv16)

def discriminator(image, batch_size, reuse=False, train=True):

   #if reuse:
   #   print 'Reusing variables'
   #   tf.get_variable_scope().reuse_variables()

   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      print 'd_image:',image
      conv1 = slim.convolution(image, 64, 5, stride=2, activation_fn=None, scope='d_conv1')
      conv1 = lrelu(conv1)
      print 'd_conv1:',conv1

      conv2 = slim.convolution(conv1, 128, 5, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv2')
      conv2 = lrelu(conv2)
      print 'd_conv2:',conv2

      conv3 = slim.convolution(conv2, 256, 5, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv3')
      conv3 = lrelu(conv3)
      print 'd_conv3:',conv3
      
      conv4 = slim.convolution(conv3, 512, 3, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv4')
      conv4 = lrelu(conv4)
      print 'd_conv4:',conv4
      
      conv5 = slim.convolution(conv4, 128, 2, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv5')
      conv5 = lrelu(conv5)
      print 'd_conv5:',conv5

      conv6_flat = slim.fully_connected(tf.reshape(conv5, [batch_size, -1]), 1, activation_fn=None, scope='d_lin')

      print 'd_conv6_flat:', conv6_flat

      return tf.nn.sigmoid(conv6_flat), conv6_flat


