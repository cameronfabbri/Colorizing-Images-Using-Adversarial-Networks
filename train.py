import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'config/')

import data_ops

if __name__ == '__main__':

   '''
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

   LOSS_METHOD     = config.LOSS_METHOD
   ARCHITECTURE    = config.ARCHITECTURE
   DATASET         = config.DATASET
   CHECKPOINT_DIR  = 'checkpoints/'+LOSS_METHOD+'_'+DATASET+'_'+ARCHITECTURE+'/'
   BATCH_SIZE      = config.BATCH_SIZE
   DATA_DIR        = config.DATA_DIR
   IMAGES_DIR      = CHECKPOINT_DIR+'images/'
   PRETRAIN_EPOCHS = config.PRETRAIN_EPOCHS
   '''

   parser = argparse.ArgumentParser()
   parser.add_argument('--PRETRAIN_EPOCHS',required=True,help='Number of epochs to pretrain', type=int)
   parser.add_argument('--GAN_EPOCHS',     required=True,help='Number of epochs for GAN', type=int)
   parser.add_argument('--ARCHITECTURE',   required=True,help='Architecture for the generator')
   parser.add_argument('--DATASET',        required=True,help='The dataset to use')
   parser.add_argument('--PRETRAIN_LR',    required=True,help='Learning rate for the pretrained network')
   parser.add_argument('--GAN_LR',         required=False,default=2e-5,help='Learning rate for the GAN')
   parser.add_argument('--MULTI_GPU',      required=False,default=True,help='Use multiple GPUs or not')
   parser.add_argument('--LOSS_METHOD',    required=False,default='wasserstein',help='Loss function for GAN',
      choices=['wasserstein','least_squares','energy'])
   a = parser.parse_args()

   PRETRAIN_EPOCHS = a.PRETRAIN_EPOCHS
   GAN_EPOCHS      = a.GAN_EPOCHS
   ARCHITECTURE    = a.ARCHITECTURE
   DATASET         = a.DATASET
   PRETRAIN_LR     = a.PRETRAIN_LR
   GAN_LR          = a.GAN_LR
   MULTI_GPU       = a.MULTI_GPU
   LOSS_METHOD     = a.LOSS_METHOD

   EXPERIMENT_DIR = 'checkpoints/'+ARCHITECTURE+'_'+DATASET+'_'+LOSS_METHOD+'_'+str(PRETRAIN_EPOCHS)+'_'+str(GAN_EPOCHS)+'_'+str(PRETRAIN_LR)
   print EXPERIMENT_DIR
   exit()

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(CHECKPOINT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   global_step = tf.Variable(0, name='global_step', trainable=False)
  
   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE)
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets
   
   if ARCHITECTURE == 'pix2pix':
      import pix2pix
      encoded, conv7, conv6, conv5, conv4, conv3, conv2, conv1 = netG_encoder(L_image)
      decoded = netG_decoder(encoded, conv7, conv6, conv5, conv4, conv3, conv2, conv1)
      # encode L and decode to ab -> this should be in [-1, 1] range
      enc_test_images, tconv7, tconv6, tconv5, tconv4, tconv3, tconv2, tconv1 = netG_encoder(test_L)
      dec_test_images = netG_decoder(enc_test_images, tconv7, tconv6, tconv5, tconv4, tconv3, tconv2, tconv1)
      colored_image   = tf.concat([test_L, dec_test_images], axis=3)
      
      # find L1 loss of decoded and original -> this loss is combined with D loss
      l1_loss = tf.reduce_mean(tf.abs(decoded-ab_image))
   
      # weight of how much the l1 loss takes into account 
      l1_weight = 100.0
   
      # total error for the critic
      errD = tf.reduce_mean(errD_real - errD_fake)
      # error for the generator, including the L1 loss
      errG = tf.reduce_mean(errD_fake) + l1_loss*l1_weight
      tf.summary.scalar('encoding_loss', l1_loss)
      
   if ARCHITECTURE == 'colorarch':
      import colorarch
      # generate a colored image
      gen_img = colorarch.netG(L_image, BATCH_SIZE)

      # send real image to D
      errD_real = colorarch.netD(ab_image, BATCH_SIZE)

      # send generated image to D
      errD_fake = colorarch.netD(gen_img, BATCH_SIZE, reuse=True)
  
   if LOSS_METHOD == 'wasserstein':
      errD = tf.reduce_mean(errD_real - errD_fake)
      errG = tf.reduce_mean(errD_fake) + tf.reduce_mean((ab_image-gen_img)**2)
   if LOSS_METHOD == 'energy':
      print 'using ebgans'

   if LOSS_METHOD == 'least_squares':
      errD = tf.reduce_mean((errD_real-b)**2 - (errD_fake-a)**2)
      errG = tf.reduce_mean((errD_fake-c)**2)

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # MSE loss for pretraining
   if PRETRAIN_EPOCHS > 0:
      print 'Pretraining generator...'
      mse_loss = tf.reduce_mean((ab_image-gen_img)**2)
      mse_train_op = tf.train.AdamOptimizer(LEARNING_RATE=1e-4).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      tf.add_to_collection('vars', mse_train_op)
      tf.summary.scalar('mse_loss', mse_loss)

   # clip weights in D
   clip_values = [-0.005, 0.005]
   clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(LEARNING_RATE=LEARNING_RATE).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(LEARNING_RATE=LEARNING_RATE).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(CHECKPOINT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   # only keep one model
   ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
   # restore previous model if there is one
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   
   
   ########################################### training portion
   step = sess.run(global_step)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()

   while True:
      # if PRETRAIN, don't run G or D until number of epochs is met
      epoch_num = step/(num_train/BATCH_SIZE)
      
      if PRETRAIN:
         while epoch_num < PRETRAIN_epochs:
            epoch_num = step/(num_train/BATCH_SIZE)
            s = time.time()
            sess.run(mse_train_op)
            mse, summary = sess.run([mse_loss, merged_summary_op])
            step += 1
            summary_writer.add_summary(summary, step)
            print 'step:',step,'mse:',mse,'time:',time.time()-s
            if step % 500 == 0:
               saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
               saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         PRETRAIN = False
         print 'Done PRETRAINing....training D and G now' 

      s = time.time()
      if step < 25 or step % 500 == 0:
         n_critic = 100
      else: n_critic = 15

      for critic_itr in range(n_critic):
         sess.run(D_train_op)
         sess.run(clip_discriminator_var_op)
     
      sess.run(G_train_op)
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

      summary_writer.add_summary(summary, step)
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,' time:',time.time()-s
      step += 1
      
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, CHECKPOINT_DIR+'checkpoint-'+str(step))
         saver.export_meta_graph(CHECKPOINT_DIR+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'
