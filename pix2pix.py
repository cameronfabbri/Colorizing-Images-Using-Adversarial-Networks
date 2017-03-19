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
            #output = conv2d(L_images, ngf, stride=2)
            output = slim.conv2d(L_images, 64, 4, stride=2, activation_fn=None, padding='VALID')
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
               #convolved = conv2d(rectified, out_channels, stride=2)
               #output = batchnorm(convolved)
               output = slim.conv2d(rectified, out_channels, 4, stride=2, normalizer_fn=slim.batch_norm, padding='VALID', activation_fn=None)
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
               #output = deconv(rectified, out_channels)
               #output = batchnorm(output)
               output = slim.convolution2d_transpose(rectified, out_channels, 4, stride=2, normalizer_fn=slim.batch_norm, padding='SAME', activation_fn=None)

               if dropout > 0.0: output = tf.nn.dropout(output, keep_prob=1 - dropout)
               
               layers.append(output)
         
         # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
         with tf.variable_scope('g_dec1'):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            #output = deconv(rectified, 2)
            #output = tf.tanh(output)
            output = slim.convolution2d_transpose(rectified, 2, 4, stride=2, padding='SAME', activation_fn=None)
            output = tf.tanh(output)
            layers.append(output)
            print output
   
   return output



def netD(L_images, ab_images, num_gpu, reuse=False):
   ndf = 64
   n_layers = 3
   layers = []
   print
   print 'netD'
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
            print 'L_images:',L_images
            print 'ab_images:',ab_images
            print 'input:',input
            # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            with tf.variable_scope('d_1'):
               #convolved = conv2d(input, ndf, stride=2)
               convolved = slim.conv2d(input, ndf, 4, stride=2, padding='VALID', activation_fn=None)
               rectified = lrelu(convolved, 0.2)
               layers.append(rectified)
               print rectified
               tf.add_to_collection('vars',rectified)

            # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
            for i in range(n_layers):
               with tf.variable_scope('d_%d' % (len(layers) + 1)):
                  out_channels = ndf * min(2**(i+1), 8)
                  stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
                  #convolved = conv2d(layers[-1], out_channels, stride=stride)
                  #normalized = batchnorm(convolved)
                  normalized = slim.conv2d(layers[-1], out_channels, 4, stride=stride, normalizer_fn=slim.batch_norm, activation_fn=None)
                  rectified = lrelu(normalized, 0.2)
                  layers.append(rectified)
                  print rectified
                  tf.add_to_collection('vars',rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope('d_%d' % (len(layers) + 1)):
               #output = conv2d(rectified, out_channels=1, stride=1)
               output = slim.conv2d(rectified, 1, 4, stride=1, padding='VALID', activation_fn=None)
               layers.append(output)

            tf.add_to_collection('vars',output)
            print output
            return output

