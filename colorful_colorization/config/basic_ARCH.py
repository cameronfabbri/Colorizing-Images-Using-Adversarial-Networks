import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

#Leaky RELU : https://arxiv.org/pdf/1502.01852.pdf
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)


def netG(L_image, batch_size, num_gpu):
   if   num_gpu == 0: Machines = ['/cpu:0']
   elif num_gpu == 1: Machines = ['/gpu:0']
   elif num_gpu == 2: Machines = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: Machines = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: Machines = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   for d in Machines:
      with tf.device(d):
	  conv1 = slim.convolution(L_image, 130, 3, stride=1, activation_fn=tf.identity, scope='g_conv1')
	  conv1 = lrelu(conv1)
	  
	  conv2 = slim.convolution(conv1, 66, 3, stride=1, normalizer_fn=slim.batch_norm, 
		        activation_fn=tf.identity, scope='g_conv2')
	  conv2 = lrelu(conv2)
	  
	  conv3 = slim.convolution(conv2, 65, 3, stride=1, normalizer_fn=slim.batch_norm, 
		        activation_fn=tf.identity, scope='g_conv3')
	  conv3 = lrelu(conv3)
	  
	  conv4 = slim.convolution(conv3, 65, 1, stride=1, normalizer_fn=slim.batch_norm, 
		        activation_fn=tf.identity, scope='g_conv4')
	  conv4 = lrelu(conv4)      
	  
	  conv5 = slim.convolution(conv4, 33, 3, stride=1, normalizer_fn=slim.batch_norm, 
		        activation_fn=tf.identity, scope='g_conv5')
	  conv5 = lrelu(conv5)
	  
	  conv6 = slim.convolution(conv4, 2, 3, stride=1, normalizer_fn=slim.batch_norm, 
		        activation_fn=tf.identity, scope='g_conv6')
	  conv6 = tf.nn.tanh(conv6)
	        
	       
   print '< GEN >'
   print 'conv1:',conv1
   print 'conv2:',conv2
   print 'conv3:',conv3
   print 'conv4:',conv4
   print 'conv5:',conv5
   print 'conv6:',conv6
   print '< END GEN >'
   print

   tf.add_to_collection('vars', conv1)
   tf.add_to_collection('vars', conv2)
   tf.add_to_collection('vars', conv3)
   tf.add_to_collection('vars', conv4)
   tf.add_to_collection('vars', conv5)
   tf.add_to_collection('vars', conv6)
   
   return conv6

