import tensorflow as tf
import numpy as np
import random
import ntpath
import sys
import cv2
import os
import time

from scipy import misc
from skimage import color

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'config/')

import data_ops

if __name__ == '__main__':
   
   global_step = tf.Variable(0, name='global_step', trainable=False)

   if len(sys.argv) < 2:
      print 'You must provide a config file'
      exit()

   try:
      config_file = ntpath.basename(sys.argv[1]).split('.py')[0]
      config = __import__(config_file)
   except:
      print 'config',sys.argv[1],'not found'
      print
      exit()

   loss_method    = config.loss_method
   architecture   = config.architecture
   dataset        = config.dataset
   checkpoint_dir = 'checkpoints/'+loss_method+'_'+dataset+'_'+architecture+'/'
   learning_rate  = config.learning_rate
   batch_size     = config.batch_size
   data_dir       = config.data_dir
   images_dir     = checkpoint_dir+'images/'

   Data = data_ops.loadData(data_dir, dataset, batch_size)
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   test_L = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets
   if architecture == 'pix2pix':
      import pix2pix
      enc_test_images, tconv7, tconv6, tconv5, tconv4, tconv3, tconv2, tconv1 = netG_encoder(test_L)
      dec_test_images = netG_decoder(enc_test_images, tconv7, tconv6, tconv5, tconv4, tconv3, tconv2, tconv1)
   if architecture == 'colorarch':
      import colorarch
      predict_ab = colorarch.netG(test_L, batch_size)
  
   # reconstruct prediction image from test_L and predict_ab
   prediction = data_ops.augment(predict_ab, test_L)
   prediction = tf.image.convert_image_dtype(prediction, dtype=tf.uint8, saturate=True)
   
   # reconstruct original image from test_L and ab_image
   true_image = data_ops.augment(ab_image, test_L)
   true_image = tf.image.convert_image_dtype(true_image, dtype=tf.uint8, saturate=True)
   
   saver = tf.train.Saver()
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         raise
         print "Could not restore model"
         exit()
   
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   # get predictions and true images
   colored = sess.run(prediction)
   true_   = sess.run(true_image)
   
   step = sess.run(global_step)

   # save out both
   i = 0
   for c in colored:
      misc.imsave(images_dir+str(step)+'_'+str(i)+'_col.png', c)
      if i == 3: break
      i += 1
   i = 0
   for t in true_:
      misc.imsave(images_dir+str(step)+'_'+str(i)+'_true.png', t)
      if i == 3: break
      i += 1

   exit()
