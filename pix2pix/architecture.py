import tensorflow as tf
import tensorflow.contrib.slim as slim
import sys
import config

multi_gpu = True

batch_size = config.batch_size

'''
   Leaky RELU
   https://arxiv.org/pdf/1502.01852.pdf
'''
def lrelu(x, leak=0.2, name='lrelu'):
   return tf.maximum(leak*x, x)

# should also pass in labels if we have em
def netG_encoder(L_image):

   print 'GENERATOR'
   print 'images:',L_image 
   
   ####### encoder ########
   # no batch norm on first layer

   #with tf.device('/gpu:0'):
   if 1:
      conv1 = slim.convolution(L_image, 64, 4, stride=2, activation_fn=tf.identity, scope='g_e_conv1')
      conv1 = lrelu(conv1)
      print 'conv1:',conv1
      
      conv2 = slim.convolution(conv1, 128, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv2')
      conv2 = lrelu(conv2)
      print 'conv2:',conv2
      
      conv3 = slim.convolution(conv2, 256, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv3')
      conv3 = lrelu(conv3)
      print 'conv3:',conv3

      conv4 = slim.convolution(conv3, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv4')
      conv4 = lrelu(conv4)
      print 'conv4:',conv4

      conv5 = slim.convolution(conv4, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv5')
      conv5 = lrelu(conv5)
      print 'conv5:',conv5

      conv6 = slim.convolution(conv5, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv6')
      conv6 = lrelu(conv6)
      print 'conv6:',conv6

      conv7 = slim.convolution(conv6, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv7')
      conv7 = lrelu(conv7)
      print 'conv7:',conv7

      conv8 = slim.convolution(conv7, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_e_conv8')
      conv8 = lrelu(conv8)
      print 'conv8:',conv8
      print

   tf.add_to_collection('vars',conv1)
   tf.add_to_collection('vars',conv2)
   tf.add_to_collection('vars',conv3)
   tf.add_to_collection('vars',conv4)
   tf.add_to_collection('vars',conv5)
   tf.add_to_collection('vars',conv6)
   tf.add_to_collection('vars',conv7)
   tf.add_to_collection('vars',conv8)

   return conv8, conv7, conv6, conv5, conv4, conv3, conv2, conv1
   ####### END #######


'''
   Decoder portion of the generator
'''
def netG_decoder(conv8, conv7, conv6, conv5, conv4, conv3, conv2, conv1):

   if multi_gpu: gpu_num = 1
   else: gpu_num = 0
   #with tf.device('/gpu:'+str(gpu_num)):
   if 1:
      ###### decoder ######
      dconv1 = slim.convolution2d_transpose(conv8, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv1')
      dconv1 = tf.nn.relu(dconv1)
      dconv1 = tf.nn.dropout(dconv1, 0.5)
      dconv1 = tf.concat([conv7, dconv1], 3)
      print 'dconv1:',dconv1

      dconv2 = slim.convolution2d_transpose(dconv1, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv2')
      dconv2 = tf.nn.relu(dconv2)
      dconv2 = tf.concat([conv6, dconv2], 3)
      dconv2 = tf.nn.dropout(dconv2, 0.5)
      print 'dconv2:',dconv2
      
      dconv3 = slim.convolution2d_transpose(dconv2, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv3')
      dconv3 = tf.concat([conv5, dconv3], 3)
      dconv3 = tf.nn.relu(dconv3)
      dconv3 = tf.nn.dropout(dconv3, 0.5)
      print 'dconv3:',dconv3
     
      dconv4 = slim.convolution2d_transpose(dconv3, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv4')
      dconv4 = tf.concat([conv4, dconv4], 3)
      dconv4 = tf.nn.relu(dconv4)
      print 'dconv4:',dconv4
   
   if multi_gpu: gpu_num = 2
   else: gpu_num = 0
   #with tf.device('/gpu:'+str(gpu_num)):
   if 1:
      dconv5 = slim.convolution2d_transpose(dconv4, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv5')
      dconv5 = tf.concat([conv3, dconv5], 3)
      dconv5 = tf.nn.relu(dconv5)
      print 'dconv5:',dconv5

      dconv6 = slim.convolution2d_transpose(dconv5, 256, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv6')
      dconv6 = tf.concat([conv2, dconv6], 3)
      dconv6 = tf.nn.relu(dconv6)
      print 'dconv6:',dconv6
      
      dconv7 = slim.convolution2d_transpose(dconv6, 128, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv7')
      dconv7 = tf.nn.relu(dconv7)
      dconv7 = tf.concat([conv1, dconv7], 3)
      print 'dconv7:',dconv7
      
      dconv8 = slim.convolution2d_transpose(dconv7, 64, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv8')
      dconv8 = tf.nn.relu(dconv8)
      print 'dconv8:',dconv8
      
      # return 2 channels instead of 3 because of a b colorspace
      conv9 = slim.convolution(dconv8, 2, 4, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_conv9')
      
      conv9 = tf.nn.tanh(conv9)
      
      print 'conv9:', conv9
      
   tf.add_to_collection('vars',dconv1)
   tf.add_to_collection('vars',dconv2)
   tf.add_to_collection('vars',dconv3)
   tf.add_to_collection('vars',dconv4)
   tf.add_to_collection('vars',dconv5)
   tf.add_to_collection('vars',dconv6)
   tf.add_to_collection('vars',dconv7)
   tf.add_to_collection('vars',dconv8)
   tf.add_to_collection('vars',conv9)

   print
   print 'END G'
   print
   return conv9



'''
   Discriminator network
'''
def netD(input_images, reuse=False):

   print 'DISCRIMINATOR' 
   sc = tf.get_variable_scope()
   with tf.variable_scope(sc, reuse=reuse):
      if multi_gpu: gpu_num = 3
      else: gpu_num = 0
      #with tf.device('/gpu:'+str(gpu_num)):
      if 1:
         print 'input images:',input_images
         conv1 = slim.convolution(input_images, 64, 5, stride=2, activation_fn=tf.identity, scope='d_conv1')
         conv1 = lrelu(conv1)
         print 'conv1:',conv1

         conv2 = slim.convolution(conv1, 128, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv2')
         conv2 = lrelu(conv2)
         print 'conv2:',conv2
         
         conv3 = slim.convolution(conv2, 256, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv3')
         conv3 = lrelu(conv3)
         print 'conv3:',conv3

         conv4 = slim.convolution(conv3, 512, 5, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='d_conv4')
         conv4 = lrelu(conv4)
         print 'conv4:',conv4

         conv5 = slim.convolution(conv4, 1, 4, stride=2, activation_fn=tf.identity, scope='d_conv5')
         print 'conv5:',conv5
         
         print 'END D\n'
   
      tf.add_to_collection('vars',conv1)
      tf.add_to_collection('vars',conv2)
      tf.add_to_collection('vars',conv3)
      tf.add_to_collection('vars',conv4)
      tf.add_to_collection('vars',conv5)

      return conv5
