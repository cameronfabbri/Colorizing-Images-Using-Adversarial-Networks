import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys


'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

'''
'''
def netG(L_images, z, num_gpu):
   if num_gpu == 0: gpus = ['/cpu:0']
   elif num_gpu == 1: gpus = ['/gpu:0']
   elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
   
   input_images = tf.concat([L_images, z], axis=3)

   for d in gpus:
      with tf.device(d):
         conv1 = slim.conv2d(input_images, 128, 3, stride=1, activation_fn=tf.identity, scope='g_conv1')
         conv1 = tf.concat([conv1, L_images], axis=3)
         conv1 = tf.concat([conv1, z], axis=3)
         conv1 = lrelu(conv1)

         conv2 = slim.conv2d(conv1, 64, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv2')
         conv2 = tf.concat([conv2, L_images], axis=3)
         conv2 = tf.concat([conv2, z], axis=3)
         conv2 = tf.nn.relu(conv2)

         conv3 = slim.conv2d(conv2, 64, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv3')
         conv3 = tf.concat([conv3, L_images], axis=3)
         conv3 = tf.nn.relu(conv3)

         conv4 = slim.conv2d(conv3, 64, 1, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv4')
         conv4 = tf.concat([conv4, L_images], axis=3)
         conv4 = tf.nn.relu(conv4)
         
         conv5 = slim.conv2d(conv4, 32, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv5')
         conv5 = tf.concat([conv5, L_images], axis=3)
         conv5 = lrelu(conv5)
   
         conv6 = slim.conv2d(conv4, 2, 3, stride=1, activation_fn=tf.identity, scope='g_conv6')
         conv6 = tf.nn.tanh(conv6)
   
   print 'GENERATOR'
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'conv5:',conv5
   print 'conv6:',conv6
   print 'END G'
   print
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', conv6)
   return conv6


'''
   Discriminator network
'''
def netD(L_images, ab_images, num_gpu, loss, reuse=False):
   
   input_images = tf.concat([L_images, ab_images], axis=3)
   
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      if num_gpu == 0: gpus = ['/cpu:0']
      elif num_gpu == 1: gpus = ['/gpu:0']
      elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
      elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
      elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
      
      for d in gpus:
         with tf.device(d):
            conv1 = slim.conv2d(input_images, 64, 5, stride=2, activation_fn=tf.identity, scope='d_conv1')
            conv1 = lrelu(conv1)

            conv2 = slim.conv2d(conv1, 128, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv2')
            conv2 = lrelu(conv2)
         
            conv3 = slim.conv2d(conv2, 256, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv3')
            conv3 = lrelu(conv3)

            conv4 = slim.conv2d(conv3, 512, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv4')
            conv4 = lrelu(conv4)

            if loss == 'gan':
               conv4_flat = slim.flatten(conv4)
               output = slim.fully_connected(conv4_flat, 1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_fc')
               output = lrelu(output)
            elif loss == 'wasserstein':
               output = slim.conv2d(conv4, 1, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv5')
               output = lrelu(output)

   print 'DISCRIMINATOR' 
   print 'input images:',input_images
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'output:',output
   print 'END D\n'
   
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', output)
   
   return output
