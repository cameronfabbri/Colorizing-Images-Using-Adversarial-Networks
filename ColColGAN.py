import tensorflow as tf
import tensorflow.contrib.layers as layers
import sys

sys.path.insert(0, 'ops/')
from tf_ops import lrelu, conv2d, batch_norm, conv2d_transpose, relu, tanh

def netG(L_image, num_gpu):
   if   num_gpu == 0: Machines = ['/cpu:0']
   elif num_gpu == 1: Machines = ['/gpu:0']
   elif num_gpu == 2: Machines = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: Machines = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: Machines = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   for d in Machines:
      with tf.device(d):
	  with tf.variable_scope('g_conv1_1'):
	        conv1_1 = conv2d(L_image, 64, kernel_size=3, stride=1)
	        conv1_1 = tf.nn.relu(conv1_1)
	  with tf.variable_scope('g_conv1_2'):
	        conv1_2 = conv2d(conv1_1, 64, kernel_size=3, stride=2)
	        conv1_2 = tf.nn.relu(conv1_2)
	        conv1_2 = batch_norm(conv1_2)
	  with tf.variable_scope('g_conv2_1'):
	        conv2_1 = conv2d(conv1_2, 128, kernel_size=3, stride=1)
	        conv2_1 = tf.nn.relu(conv2_1)
	  with tf.variable_scope('g_conv2_2'):
	        conv2_2 = conv2d(conv2_1, 128, kernel_size=3, stride=2)
	        conv2_2 = tf.nn.relu(conv2_2)
	        conv2_2 = batch_norm(conv2_2)
	  with tf.variable_scope('g_conv3_1'):
	        conv3_1 = conv2d(conv2_2, 256, kernel_size=3, stride=1)
	        conv3_1 = tf.nn.relu(conv3_1)
	  with tf.variable_scope('g_conv3_2'):
	        conv3_2 = conv2d(conv3_1, 256, kernel_size=3, stride=1)
	        conv3_2 = tf.nn.relu(conv3_2)
	  with tf.variable_scope('g_conv3_3'):
	        conv3_3 = conv2d(conv3_2, 256, kernel_size=3, stride=2)
	        conv3_3 = tf.nn.relu(conv3_3)
	        conv3_3 = batch_norm(conv3_3)      
	  with tf.variable_scope('g_conv4_1'):
	        conv4_1 = conv2d(conv3_3, 512, kernel_size=3, stride=1)
	        conv4_1 = tf.nn.relu(conv4_1)
	  with tf.variable_scope('g_conv4_2'):
	        conv4_2 = conv2d(conv4_1, 512, kernel_size=3, stride=1)
	        conv4_2 = tf.nn.relu(conv4_2)
	  with tf.variable_scope('g_conv4_3'):
	        conv4_3 = conv2d(conv4_2, 512, kernel_size=3, stride=1)
	        conv4_3 = tf.nn.relu(conv4_3)
	        conv4_3 = batch_norm(conv4_3)   
	  with tf.variable_scope('g_conv5_1'):
	        conv5_1 = conv2d(conv4_3, 512, kernel_size=3, stride=1)
	        conv5_1 = tf.nn.relu(conv5_1)
	  with tf.variable_scope('g_conv5_2'):
	        conv5_2 = conv2d(conv5_1, 512, kernel_size=3, stride=1)
	        conv5_2 = tf.nn.relu(conv5_2)
	  with tf.variable_scope('g_conv5_3'):
	        conv5_3 = conv2d(conv5_2, 512, kernel_size=3, stride=1)
	        conv5_3 = tf.nn.relu(conv5_3)
	        conv5_3 = batch_norm(conv5_3)
	  with tf.variable_scope('g_conv6_1'):
	        conv6_1 = conv2d(conv5_3, 512, kernel_size=3, stride=1)
	        conv6_1 = tf.nn.relu(conv6_1)
	  with tf.variable_scope('g_conv6_2'):
	        conv6_2 = conv2d(conv6_1, 512, kernel_size=3, stride=1)
	        conv6_2 = tf.nn.relu(conv6_2)
	  with tf.variable_scope('g_conv6_3'):
	        conv6_3 = conv2d(conv6_2, 512, kernel_size=3, stride=1)
	        conv6_3 = tf.nn.relu(conv6_3)
	        conv6_3 = batch_norm(conv6_3)
	  with tf.variable_scope('g_conv7_1'):
	        conv7_1 = conv2d(conv6_3, 256, kernel_size=3, stride=1)
	        conv7_1 = tf.nn.relu(conv7_1)
	  with tf.variable_scope('g_conv7_2'):
	        conv7_2 = conv2d(conv7_1, 256, kernel_size=3, stride=1)
	        conv7_2 = tf.nn.relu(conv7_2)
	  with tf.variable_scope('g_conv7_3'):
	        conv7_3 = conv2d(conv6_2, 256, kernel_size=3, stride=1)
	        conv7_3 = tf.nn.relu(conv7_3)
	        conv7_3 = batch_norm(conv7_3)
	  with tf.variable_scope('g_conv8_1'):
	        conv8_1 = conv2d_transpose(conv7_3, 256, kernel_size=3, stride=2)
	        conv8_1 = tf.nn.relu(conv8_1)
	  
	  with tf.variable_scope('g_conv8_2'):
	        conv8_2 = conv2d(conv8_1, 128, kernel_size=3, stride=1)
	        conv8_2 = tf.nn.relu(conv8_2)  
	  with tf.variable_scope('g_conv8_3'):
	        conv8_3 = conv2d(conv8_2, 128, kernel_size=3, stride=1)
	        conv8_3 = tf.nn.relu(conv8_3)
	        conv8_3 = batch_norm(conv8_3)
	  
	  
	  with tf.variable_scope('g_conv9_1'):
	        conv9_1 = conv2d_transpose(conv8_3, 64, stride=2, kernel_size=4)
	        conv9_1 = tf.nn.relu(conv9_1) 
	  with tf.variable_scope('g_conv9_2'):
	        conv9_2 = conv2d_transpose(conv9_1, 2, stride=2, kernel_size=4)
	        conv9_2 = tf.nn.tanh(conv9_2)
	        
      return conv9_2


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

            print L_images
            print ab_images
            print

            input_images = tf.concat([L_images, ab_images], axis=3)

            # trying the pixel gan architecture
            with tf.variable_scope('d_conv1'): conv1 = lrelu(conv2d(input_images, 64, kernel_size=4, stride=2))
            with tf.variable_scope('d_conv2'): conv2 = lrelu(batch_norm(conv2d(conv1, 128, kernel_size=4, stride=2)))
            with tf.variable_scope('d_conv3'): conv3 = lrelu(batch_norm(conv2d(conv2, 256, kernel_size=4, stride=2)))
            with tf.variable_scope('d_conv4'): conv4 = lrelu(batch_norm(conv2d(conv3, 512, kernel_size=4, stride=1)))
            with tf.variable_scope('d_conv5'): conv5 = conv2d(conv4, 1, stride=1)

            print conv1
            print conv2
            print conv3
            print conv4
            print conv5
            return conv5

