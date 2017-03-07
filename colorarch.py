import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

multi_gpu = True

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

'''
'''
def netG(L_image, batch_size):

   if multi_gpu: gpu_num = 2
   else: gpu_num = 0
   with tf.device('/gpu:'+str(gpu_num)):
      conv1 = slim.convolution(L_image, 64, 3, stride=2, activation_fn=tf.identity, scope='g_conv1')
      conv1 = lrelu(conv1)

      conv2 = slim.convolution(conv1, 128, 3, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv2')
      conv2 = lrelu(conv2)

      conv3 = slim.convolution(conv2, 128, 3, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv3')
      conv3 = lrelu(conv3)

      conv4 = slim.convolution(conv3, 256, 1, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv4')
      conv4 = lrelu(conv4)
      
      conv5 = slim.convolution(conv4, 256, 3, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv3')
      conv5 = lrelu(conv5)

      conv4 = slim.convolution(conv3, 512, 1, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_conv4')
      conv4 = lrelu(conv4)

   print 'GENERATOR'
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'END G'
   print
   
   exit()
   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)

   return conv5 


'''
   Discriminator network
'''
def netD(input_images, batch_size, reuse=False):
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      if multi_gpu: gpu_num = 3
      else: gpu_num = 0
      with tf.device('/gpu:'+str(gpu_num)):
         conv1 = slim.convolution(input_images, 64, 5, stride=2, activation_fn=tf.identity, scope='d_conv1')
         conv1 = lrelu(conv1)

         conv2 = slim.convolution(conv1, 128, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv2')
         conv2 = lrelu(conv2)
      
         conv3 = slim.convolution(conv2, 256, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv3')
         conv3 = lrelu(conv3)

         conv4 = slim.convolution(conv3, 512, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv4')
         conv4 = lrelu(conv4)

         conv5 = slim.convolution(conv4, 1, 4, stride=2, activation_fn=tf.identity, scope='d_conv5')
      
      print 'DISCRIMINATOR' 
      print 'input images:',input_images
      print 'conv1:',conv1
      print 'conv2:',conv2
      print 'conv3:',conv3
      print 'conv4:',conv4
      print 'conv5:',conv5
      print 'END D\n'
      
      tf.add_to_collection('vars', conv1)
      tf.add_to_collection('vars', conv2)
      tf.add_to_collection('vars', conv3)
      tf.add_to_collection('vars', conv4)
      tf.add_to_collection('vars', conv5)
      
      return conv5
