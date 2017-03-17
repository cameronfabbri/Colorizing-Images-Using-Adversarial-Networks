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

# should also pass in labels if we have em
def netG_encoder(L_image, num_gpu):
   if num_gpu == 0: gpus = ['/cpu:0']
   elif num_gpu == 1: gpus = ['/gpu:0']
   elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
   
   print 'GENERATOR encoder'
   print 'images:',L_image 

   for d in gpus:
      with tf.device(d):
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

   g_layers = {
      'conv1':conv1,
      'conv2':conv2,
      'conv3':conv3,
      'conv4':conv4,
      'conv5':conv5,
      'conv6':conv6,
      'conv7':conv7,
      'conv8':conv8,
   }

   return g_layers

'''
   Decoder portion of the generator
'''
def netG_decoder(g_layers, num_gpu):

   conv1 = g_layers['conv1']
   conv2 = g_layers['conv2']
   conv3 = g_layers['conv3']
   conv4 = g_layers['conv4']
   conv5 = g_layers['conv5']
   conv6 = g_layers['conv6']
   conv7 = g_layers['conv7']
   conv8 = g_layers['conv8']
   
   if num_gpu == 0: gpus = ['/cpu:0']
   elif num_gpu == 1: gpus = ['/gpu:0']
   elif num_gpu == 2: gpus = ['/gpu:0', '/gpu:1']
   elif num_gpu == 3: gpus = ['/gpu:0', '/gpu:1', '/gpu:2']
   elif num_gpu == 4: gpus = ['/gpu:0', '/gpu:1', '/gpu:2', '/gpu:3']
   
   print 'GENERATOR decoder'

   for d in gpus:
      with tf.device(d):
         ###### decoder ######
         dconv1 = slim.convolution2d_transpose(conv8, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv1')
         dconv1 = tf.nn.dropout(dconv1, 0.5)
         dconv1 = tf.nn.relu(dconv1)
         dconv1 = tf.concat([conv7, dconv1], 3)

         dconv2 = slim.convolution2d_transpose(dconv1, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv2')
         dconv2 = tf.nn.dropout(dconv2, 0.5)
         dconv2 = tf.nn.relu(dconv2)
         dconv2 = tf.concat([conv6, dconv2], 3)
         
         dconv3 = slim.convolution2d_transpose(dconv2, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv3')
         dconv3 = tf.nn.dropout(dconv3, 0.5)
         dconv3 = tf.nn.relu(dconv3)
         dconv3 = tf.concat([conv5, dconv3], 3)
        
         dconv4 = slim.convolution2d_transpose(dconv3, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv4')
         dconv4 = tf.nn.relu(dconv4)
         dconv4 = tf.concat([conv4, dconv4], 3)
      
         dconv5 = slim.convolution2d_transpose(dconv4, 512, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv5')
         dconv5 = tf.nn.relu(dconv5)
         dconv5 = tf.concat([conv3, dconv5], 3)

         dconv6 = slim.convolution2d_transpose(dconv5, 256, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv6')
         dconv6 = tf.nn.relu(dconv6)
         dconv6 = tf.concat([conv2, dconv6], 3)
         
         dconv7 = slim.convolution2d_transpose(dconv6, 128, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv7')
         dconv7 = tf.nn.relu(dconv7)
         dconv7 = tf.concat([conv1, dconv7], 3)
         
         dconv8 = slim.convolution2d_transpose(dconv7, 64, 4, stride=2, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_dconv8')
         dconv8 = tf.nn.relu(dconv8)
         
         # return 2 channels instead of 3 because of a b colorspace
         #conv9 = slim.convolution(dconv8, 2, 4, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.identity, scope='g_d_conv9')
         conv9 = slim.convolution(dconv8, 2, 4, stride=1, activation_fn=tf.identity, scope='g_d_conv9')
         conv9 = tf.nn.tanh(conv9)
         
   print 'dconv1:',dconv1
   print 'dconv2:',dconv2
   print 'dconv3:',dconv3
   print 'dconv4:',dconv4
   print 'dconv5:',dconv5
   print 'dconv6:',dconv6
   print 'dconv7:',dconv7
   print 'dconv8:',dconv8
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
def netD(ab_images, L_images, num_gpu, reuse=False):

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
