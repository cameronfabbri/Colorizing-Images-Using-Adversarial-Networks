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
#from tf_ops import fc_layer, conv2d, conv2d_transpose, lrelu, tanh, sig, batch_norm
from tf_ops import lrelu, relu

def linear(input_, output_size, scope=None, stddev=0.02, bias_start=0.0, with_w=False):
   shape = input_.get_shape().as_list()
   with tf.variable_scope(scope or "Linear"):
      matrix = tf.get_variable("Matrix", [shape[1], output_size], tf.float32, tf.random_normal_initializer(stddev=stddev))
      bias = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(bias_start))       
      return tf.matmul(input_, matrix) + bias

def generator(z, batch_size, dataset, train=True):

   print 'z:',z
   # project and reshape z
   #z = fc_layer(z, 4*4*1024, False, 'g_lin')
   z = slim.fully_connected(z, 4*4*1024, activation_fn=None, scope='g_lin')
   z = tf.reshape(z, [batch_size, 4, 4, 1024])
   print 'z:',z

   #conv_t1 = lrelu(batch_norm(conv2d_transpose(z, 5, 2, 512, 'g_conv_t1'), 'g_bn1', train=train))
   conv_t1 = slim.convolution2d_transpose(z, 512, 5, stride=2, scope='g_conv_t1', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv_t1 = relu(conv_t1)
   conv_t1 = tf.nn.dropout(conv_t1, 0.5)
   print 't_conv1:',conv_t1
   
   #conv_t2 = lrelu(batch_norm(conv2d_transpose(conv_t1, 5, 2, 256, 'g_conv_t2'), 'g_bn2', train=train))
   conv_t2 = slim.convolution2d_transpose(conv_t1, 256, 5, stride=2, scope='g_conv_t2', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv_t2 = relu(conv_t2)
   conv_t2 = tf.nn.dropout(conv_t2, 0.5)
   print 't_conv2:',conv_t2

   #conv_t3 = lrelu(batch_norm(conv2d_transpose(conv_t2, 5, 2, 128, 'g_conv_t3'), 'g_bn3', train=train))
   conv_t3 = slim.convolution2d_transpose(conv_t2, 128, 5, stride=2, scope='g_conv_t3', activation_fn=None, normalizer_fn=slim.batch_norm)
   conv_t3 = relu(conv_t3)
   conv_t3 = tf.nn.dropout(conv_t3, 0.5)
   print 't_conv3:',conv_t3
   
   if dataset == 'imagenet' or dataset == 'lsun':
      conv_t4 = relu(batch_norm(conv2d_transpose(conv_t3, 5, 2, 3, 'g_conv_t4'), 'g_bn4', train=train))
      print 't_conv4:',conv_t4

   if dataset == 'mnist':
      #conv_t4 = lrelu(batch_norm(conv2d_transpose(conv_t3, 5, 2, 1, 'g_conv_t4'), 'g_bn4', train=train))
      conv_t4 = slim.convolution2d_transpose(conv_t3, 1, 5, stride=2, scope='g_conv_t4', activation_fn=None)
      conv_t4 = conv_t4[:,:28,:28,:]
      print 't_conv4:',conv_t4

   return tf.nn.tanh(conv_t4)

def discriminator(image, batch_size, reuse=False, train=True):

   if reuse:
      print 'Reusing variables'
      tf.get_variable_scope().reuse_variables()

   print 'd_image:',image
   #conv1 = lrelu(conv2d(image, 5, 2, 64, 'd_conv1'))
   conv1 = slim.convolution(image, 64, 5, stride=2, activation_fn=None, scope='d_conv1')
   conv1 = lrelu(conv1)
   print 'd_conv1:',conv1

   #conv2 = lrelu(batch_norm(conv2d(conv1, 5, 2, 128, 'd_conv2'), 'd_bn1'))
   conv2 = slim.convolution(conv1, 128, 5, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv2')
   conv2 = lrelu(conv2)
   print 'd_conv2:',conv2

   #conv3 = lrelu(batch_norm(conv2d(conv2, 5, 2, 256, 'd_conv3'), 'd_bn2', train=train))
   conv3 = slim.convolution(conv2, 256, 5, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv3')
   conv3 = lrelu(conv3)
   print 'd_conv3:',conv3
   
   #conv4 = lrelu(batch_norm(conv2d(conv3, 5, 2, 512, 'd_conv4'), 'd_bn3', train=train))
   conv4 = slim.convolution(conv3, 512, 3, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv4')
   conv4 = lrelu(conv4)
   print 'd_conv4:',conv4

   conv5 = slim.convolution(conv4, 256, 3, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv5')
   conv5 = lrelu(conv5)
   print 'd_conv5:',conv5
   
   conv6 = slim.convolution(conv4, 1, 3, stride=2, activation_fn=None, normalizer_fn=slim.batch_norm, scope='d_conv6')
   #conv6 = lrelu(conv6)
   print 'd_conv6:',conv6

   conv6_flat = tf.reshape(conv6, [batch_size, -1])
   print 'd_conv6_flat:', conv6_flat

   return tf.nn.sigmoid(conv6_flat), conv6_flat

   #conv5 = lrelu(batch_norm(conv2d(conv4, 5, 2, 256, 'd_conv5'), 'd_bn5', train=train))

   # conv that is the same size as the feature maps with same stride size so returns (batch_size, 1, 1, 1)
   # then resized to (batch_size, 1) for the logit
   #conv5 = tf.reshape(lrelu(batch_norm(conv2d(conv4, 4, 4, 1, 'd_conv5'), 'd_bn4', train=train)), [batch_size, 1])

   # need to flatten before because after transpose conv size is lost
   #conv4_flat = tf.reshape(conv4, [batch_size, -1])

   #conv5_flat = linear(tf.reshape(conv4, [batch_size, -1]), 1, 'd_conv5_lin')
   #fc1 = batch_norm(fc_layer(conv4_flat, 1, False, 'd_fc1'), 'd_bn4', train=train)
   #print 'd_conv5:',conv5
   #return tf.nn.sigmoid(conv5_flat), conv5_flat
   # returning the decision made by D
   #return tf.nn.sigmoid(conv5)



