import tensorflow as tf
import numpy as np
import cPickle as pickle
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
   
   if len(sys.argv) < 2:
      print 'You must provide an info.pkl file'
      exit()

   global_step = tf.Variable(0, name='global_step', trainable=False)

   pkl_file = open(sys.argv[1], 'rb')
   a = pickle.load(pkl_file)

   PRETRAIN_EPOCHS = a['PRETRAIN_EPOCHS']
   GAN_EPOCHS      = a['GAN_EPOCHS']
   ARCHITECTURE    = a['ARCHITECTURE']
   DATASET         = a['DATASET']
   DATA_DIR        = a['DATA_DIR']
   PRETRAIN_LR     = a['PRETRAIN_LR']
   GAN_LR          = a['GAN_LR']
   NUM_GPU         = a['NUM_GPU']
   LOSS_METHOD     = a['LOSS_METHOD']
   NUM_CRITIC      = a['NUM_CRITIC']
   BATCH_SIZE      = a['BATCH_SIZE']
   JITTER          = a['JITTER']
   SIZE            = a['SIZE']

   EXPERIMENT_DIR = 'checkpoints/'+ARCHITECTURE+'_'+DATASET+'_'+LOSS_METHOD+'_'+str(PRETRAIN_EPOCHS)+'_'+str(GAN_EPOCHS)+'_'+str(PRETRAIN_LR)+'_'+str(NUM_CRITIC)+'_'+str(GAN_LR)+'_'+str(JITTER)+'_'+str(SIZE)+'/'
   IMAGES_DIR = EXPERIMENT_DIR+'images/'
   
   print
   print 'PRETRAIN_EPOCHS: ',PRETRAIN_EPOCHS
   print 'GAN_EPOCHS:      ',GAN_EPOCHS
   print 'ARCHITECTURE:    ',ARCHITECTURE
   print 'LOSS_METHOD:     ',LOSS_METHOD
   print 'PRETRAIN_LR:     ',PRETRAIN_LR
   print 'DATASET:         ',DATASET
   print 'GAN_LR:          ',GAN_LR
   print 'NUM_GPU:         ',NUM_GPU
   print

   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE, train=False, SIZE=SIZE)
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   test_L = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets
   if ARCHITECTURE == 'pix2pix':
      import pix2pix
      g_layers = pix2pix.netG_encoder(test_L, 0)
      predict_ab = pix2pix.netG_decoder(g_layers, 0)
   if ARCHITECTURE == 'colorarch':
      import colorarch
      predict_ab = colorarch.netG(test_L, BATCH_SIZE, 0)
   if ARCHITECTURE == 'cganarch':
      import cganarch
      z = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 64, 64, 1))
      predict_ab = cganarch.netG(test_L, z, 0)

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
   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
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

   #batch_z = np.random.normal(-1.0, 1.0, size=[BATCH_SIZE, 64, 64, 1]).astype(np.float32)
   # get predictions and true images
   #colored = sess.run(prediction, feed_dict={z:batch_z})
   colored = sess.run(prediction)
   true_   = sess.run(true_image)
   
   step = sess.run(global_step)

   # save out both
   i = 0
   for c in colored:
      misc.imsave(IMAGES_DIR+str(step)+'_'+str(i)+'_col.png', c)
      if i == 10: break
      i += 1
   i = 0
   #for t in true_:
   #   misc.imsave(IMAGES_DIR+str(step)+'_'+str(i)+'_true.png', t)
   #   #if i == 3: break
   #   i += 1

   exit()
