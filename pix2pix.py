import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, 'ops/')
from tf_ops import *

def netG(L_images, num_gpu):
   
   if num_gpu == 0: gpus = ['/cpu:0']
   elif num_gpu == 1: gpus = ['/gpu:0']
   elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   ngf = 64
   layers = []
      
   for d in gpus:
      with tf.device(d):
         # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
         with tf.variable_scope('g_enc1'):
            output = conv2d(L_images, ngf, stride=2)
            layers.append(output)
            print(output)
         
         layer_specs = [
            ngf * 2, # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            ngf * 4, # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            ngf * 8, # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            ngf * 8, # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            ngf * 8, # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            ngf * 8, # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            ngf * 8, # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
         ]

         for out_channels in layer_specs:
            with tf.variable_scope('g_enc%d' % (len(layers) + 1)):
               rectified = lrelu(layers[-1], 0.2)
               # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
               convolved = conv2d(rectified, out_channels, stride=2)
               output = batchnorm(convolved)
               layers.append(output)
               print output

         layer_specs = [
            (ngf * 8, 0.5),   # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 0.5),   # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 0.0),   # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 0.0),   # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 0.0),   # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 0.0),       # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
         ]

         num_encoder_layers = len(layers)
         for decoder_layer, (out_channels, dropout) in enumerate(layer_specs):
            skip_layer = num_encoder_layers - decoder_layer - 1
            
            with tf.variable_scope('g_dec%d' % (skip_layer + 1)):
               if decoder_layer == 0:
                  # first decoder layer doesn't have skip connections
                  # since it is directly connected to the skip_layer
                  input = layers[-1]
               else:
                  input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

               print input
               rectified = tf.nn.relu(input)
               # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
               output = deconv(rectified, out_channels)
               output = batchnorm(output)

               if dropout > 0.0: output = tf.nn.dropout(output, keep_prob=1 - dropout)
               
               layers.append(output)
         
         # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
         with tf.variable_scope('g_dec1'):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, 2)
            output = tf.tanh(output)
            layers.append(output)
            print output

   return output



def netD(L_images, ab_images, num_gpu, reuse=False):
   ndf = 64
   n_layers = 3
   layers = []
   
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      if num_gpu == 0: gpus = ['/cpu:0']
      elif num_gpu == 1: gpus = ['/gpu:0']
      elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
      elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
      elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

      for d in gpus:
         with tf.device(d):

            # 2x [batch, height, width, in_channels] => [batch, height, width, in_channels * 2]
            input = tf.concat([L_images, ab_images], axis=3)

            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope('d_1'):
               convolved = conv2d(input, ndf, stride=2)
               rectified = lrelu(convolved, 0.2)
               layers.append(rectified)

               # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
               # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
               # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
               for i in range(n_layers):
                  with tf.variable_scope('d_%d' % (len(layers) + 1)):
                     out_channels = ndf * min(2**(i+1), 8)
                     stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                     convolved = conv2d(layers[-1], out_channels, stride=stride)
                     normalized = batchnorm(convolved)
                     rectified = lrelu(normalized, 0.2)
                     layers.append(rectified)

               # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
               with tf.variable_scope('d_%d' % (len(layers) + 1)):
                  convolved = conv2d(rectified, out_channels=1, stride=1)
                  output = tf.sigmoid(convolved)
                  layers.append(output)

               print output
               return output



'''
   Discriminator network
'''
def netD_(ab_images, L_images, num_gpu, reuse=False):

   # input images are ab_images concat with L
   input_images = tf.concat([L_images, ab_images], axis=3)
   
   print 'DISCRIMINATOR' 
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      if num_gpu == 0: gpus = ['/cpu:0']
      elif num_gpu == 1: gpus = ['/gpu:0']
      elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
      elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
      elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

      for d in gpus:
         with tf.device(d):
            conv1 = slim.convolution(input_images, 64, 5, stride=2, activation_fn=tf.identity, scope='d_conv1')
            conv1 = tf.nn.relu(conv1)

            conv2 = slim.convolution(conv1, 128, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv2')
            conv2 = tf.nn.relu(conv2)
            
            conv3 = slim.convolution(conv2, 256, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv3')
            conv3 = tf.nn.relu(conv3)
            
            conv4 = slim.convolution(conv3, 512, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv4')
            conv4 = tf.nn.relu(conv4)
            
            conv5 = slim.convolution(conv4, 1, 4, stride=2, activation_fn=tf.identity, scope='d_conv5')
         
   
      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      
      tf.add_to_collection('vars',conv1)
      tf.add_to_collection('vars',conv2)
      tf.add_to_collection('vars',conv3)
      tf.add_to_collection('vars',conv4)
      tf.add_to_collection('vars',conv5)
      print 'END D\n'
      return conv5
