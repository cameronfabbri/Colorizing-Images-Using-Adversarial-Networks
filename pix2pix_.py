import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys

sys.path.insert(0, 'ops/')
from tf_ops import lrelu, conv2d, batch_norm, conv2d_transpose, relu

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
         
         layer_specs = [
            (64 * 8, 0.5),   # decoder_8: [batch, 1, 1, 64 * 8] => [batch, 2, 2, 64 * 8 * 2]
            (64 * 8, 0.5),   # decoder_7: [batch, 2, 2, 64 * 8 * 2] => [batch, 4, 4, 64 * 8 * 2]
            (64 * 8, 0.5),   # decoder_6: [batch, 4, 4, 64 * 8 * 2] => [batch, 8, 8, 64 * 8 * 2]
            (64 * 8, 0.0),   # decoder_5: [batch, 8, 8, 64 * 8 * 2] => [batch, 16, 16, 64 * 8 * 2]
            (64 * 4, 0.0),   # decoder_4: [batch, 16, 16, 64 * 8 * 2] => [batch, 32, 32, 64 * 4 * 2]
            (64 * 2, 0.0),   # decoder_3: [batch, 32, 32, 64 * 4 * 2] => [batch, 64, 64, 64 * 2 * 2]
            (64, 0.0),       # decoder_2: [batch, 64, 64, 64 * 2 * 2] => [batch, 128, 128, 64 * 2]
         ]

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
            dec_convt5 = conv2d_transpose(dec_convt5, 512, stride=2, kernel_size=4)
            dec_convt5 = batch_norm(dec_convt5)
            dec_convt5 = relu(dec_convt5)
         
         exit()
         

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
               #output = slim.convolution2d_transpose(rectified, out_channels, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity)

               if dropout > 0.0: output = tf.nn.dropout(output, keep_prob=1 - dropout)
               
               layers.append(output)
         
         # decoder_1: [batch, 128, 128, 64 * 2] => [batch, 256, 256, generator_outputs_channels]
         with tf.variable_scope('g_dec1'):
            input = tf.concat([layers[-1], layers[0]], axis=3)
            rectified = tf.nn.relu(input)
            output = deconv(rectified, 2)
            output = tf.tanh(output)
            #output = slim.convolution2d_transpose(rectified, 2, 4, stride=2, padding='SAME', activation_fn=tf.identity)
            #output = tf.tanh(output)
            layers.append(output)
            print output
   
   return layers[-1]



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
               convolved = conv2d(input, ndf, stride=2)
               #convolved = slim.conv2d(input, ndf, 4, stride=2, activation_fn=tf.identity)
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
                  convolved = conv2d(layers[-1], out_channels, stride=stride)
                  normalized = batchnorm(convolved)
                  #normalized = slim.conv2d(layers[-1], out_channels, 4, stride=stride, normalizer_fn=slim.batch_norm, activation_fn=tf.identity)
                  rectified = lrelu(normalized, 0.2)
                  layers.append(rectified)
                  print rectified
                  tf.add_to_collection('vars',rectified)

            # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
            with tf.variable_scope('d_%d' % (len(layers) + 1)):
               output = conv2d(rectified, out_channels=1, stride=1)
               #output = slim.conv2d(rectified, 1, 4, stride=1, activation_fn=tf.identity)
               layers.append(output)

            tf.add_to_collection('vars',output)
            print output
            return layers[-1]

