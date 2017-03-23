import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, 'ops/')
from tf_ops import lrelu, conv2d, batch_norm, conv2d_transpose, relu, tanh

# def conv2d(batch_input, out_channels, stride, name='conv', kernel_size=4):
def netG(L_images, num_gpu):
   
   if num_gpu == 0: gpus = ['/cpu:0']
   elif num_gpu == 1: gpus = ['/gpu:0']
   elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']

   for d in gpus:
      with tf.device(d):

         with tf.variable_scope('g_enc1'): enc_conv1 = lrelu(conv2d(L_images, 64, stride=2))
         with tf.variable_scope('g_enc2'): enc_conv2 = lrelu(batch_norm(conv2d(enc_conv1, 128, stride=2, kernel_size=4)))
         with tf.variable_scope('g_enc3'): enc_conv3 = lrelu(batch_norm(conv2d(enc_conv2, 256, stride=2, kernel_size=4)))
         with tf.variable_scope('g_enc4'): enc_conv4 = lrelu(batch_norm(conv2d(enc_conv3, 512, stride=2, kernel_size=4)))
         with tf.variable_scope('g_enc5'): enc_conv5 = lrelu(batch_norm(conv2d(enc_conv4, 512, stride=2, kernel_size=4)))
         with tf.variable_scope('g_enc6'): enc_conv6 = lrelu(batch_norm(conv2d(enc_conv5, 512, stride=2, kernel_size=4)))
         with tf.variable_scope('g_enc7'): enc_conv7 = lrelu(batch_norm(conv2d(enc_conv6, 512, stride=2, kernel_size=4)))
         with tf.variable_scope('g_enc8'): enc_conv8 = lrelu(batch_norm(conv2d(enc_conv7, 512, stride=2, kernel_size=4)))

         print 'enc_conv1:',enc_conv1
         print 'enc_conv2:',enc_conv2
         print 'enc_conv3:',enc_conv3
         print 'enc_conv4:',enc_conv4
         print 'enc_conv5:',enc_conv5
         print 'enc_conv6:',enc_conv6
         print 'enc_conv7:',enc_conv7
         print 'enc_conv8:',enc_conv8
         
         with tf.variable_scope('g_dec1'):
            dec_convt1 = conv2d_transpose(enc_conv8, 512, stride=2, kernel_size=4)
            dec_convt1 = batch_norm(dec_convt1)
            dec_convt1 = relu(dec_convt1)
            dec_convt1 = tf.nn.dropout(dec_convt1, keep_prob=0.5)
            print dec_convt1
         with tf.variable_scope('g_dec2'):
            dec_convt2 = tf.concat([dec_convt1, enc_conv7], axis=3)
            print dec_convt2
            dec_convt2 = conv2d_transpose(dec_convt2, 512, stride=2, kernel_size=4)
            dec_convt2 = batch_norm(dec_convt2)
            dec_convt2 = relu(dec_convt2)
         with tf.variable_scope('g_dec3'):
            dec_convt3 = tf.concat([enc_conv6, dec_convt2], axis=3)
            print dec_convt3
            dec_convt3 = conv2d_transpose(dec_convt3, 512, stride=2, kernel_size=4)
            dec_convt3 = batch_norm(dec_convt3)
            dec_convt3 = relu(dec_convt3)
         with tf.variable_scope('g_dec4'):
            dec_convt4 = tf.concat([enc_conv5, dec_convt3], axis=3)
            print dec_convt4
            dec_convt4 = conv2d_transpose(dec_convt4, 512, stride=2, kernel_size=4)
            dec_convt4 = batch_norm(dec_convt4)
            dec_convt4 = relu(dec_convt4)
         with tf.variable_scope('g_dec5'):
            dec_convt5 = tf.concat([enc_conv4, dec_convt4], axis=3)
            print dec_convt5
            dec_convt5 = conv2d_transpose(dec_convt5, 256, stride=2, kernel_size=4)
            dec_convt5 = batch_norm(dec_convt5)
            dec_convt5 = relu(dec_convt5)
         with tf.variable_scope('g_dec6'):
            dec_convt6 = tf.concat([enc_conv3, dec_convt5], axis=3)
            print dec_convt6
            dec_convt6 = conv2d_transpose(dec_convt6, 128, stride=2, kernel_size=4)
            dec_convt6 = batch_norm(dec_convt6)
            dec_convt6 = relu(dec_convt6)
         with tf.variable_scope('g_dec7'):
            dec_convt7 = tf.concat([enc_conv2, dec_convt6], axis=3)
            print dec_convt7
            dec_convt7 = conv2d_transpose(dec_convt7, 128, stride=2, kernel_size=4)
            dec_convt7 = batch_norm(dec_convt7)
            dec_convt7 = relu(dec_convt7)

         # output layer - ab channels
         with tf.variable_scope('g_dec8'):
            dec_convt8 = conv2d_transpose(dec_convt7, 2, stride=2, kernel_size=4)
            dec_convt8 = tanh(dec_convt8)
            
         print dec_convt8
         
   return dec_convt8


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

            input_images = tf.concat([L_images, ab_images], axis=3)

            # trying the pixel gan architecture
            with tf.variable_scope('d_conv1'): conv1 = lrelu(conv2d(input_images, 64, stride=1, kernel_size=1))
            with tf.variable_scope('d_conv2'): conv2 = lrelu(batch_norm(conv2d(conv1, 128, stride=1, kernel_size=1)))
            with tf.variable_scope('d_conv3'): conv3 = conv2d(conv1, 1, stride=1, kernel_size=1)

            print conv1
            print conv2
            print conv3

            return conv3

