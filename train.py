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
   pretrain       = config.pretrain
   pretrain_epochs = config.pretrain_epochs

   batch_size = 8

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(images_dir)
   except: pass
   
   global_step = tf.Variable(0, name='global_step', trainable=False)
  
   Data = data_ops.loadData(data_dir, dataset, batch_size)
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets
   
   if architecture == 'pix2pix':
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
      
   if architecture == 'colorarch':
      import colorarch
      # generate a colored image
      gen_img = colorarch.netG(L_image, batch_size)

      # send real image to D
      errD_real = colorarch.netD(ab_image, batch_size)

      # send generated image to D
      errD_fake = colorarch.netD(gen_img, batch_size, reuse=True)
  
   if loss_method == 'wasserstein':
      errD = tf.reduce_mean(errD_real - errD_fake)
      errG = tf.reduce_mean(errD_fake) + tf.reduce_mean((ab_image-gen_img)**2)
   if loss_method == 'energy':
      print 'using ebgans'

   if loss_method == 'least_squares':
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
   if pretrain:
      print 'Pretraining generator...'
      mse_loss = tf.reduce_mean((ab_image-gen_img)**2)
      mse_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      tf.add_to_collection('vars', mse_train_op)
      tf.summary.scalar('mse_loss', mse_loss)

   # clip weights in D
   clip_values = [-0.005, 0.005]
   clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=learning_rate).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   # only keep one model
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
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
      # if pretrain, don't run G or D until number of epochs is met
      epoch_num = step/(num_train/batch_size)
      
      if pretrain:
         while epoch_num < pretrain_epochs:
            epoch_num = step/(num_train/batch_size)
            s = time.time()
            sess.run(mse_train_op)
            mse, summary = sess.run([mse_loss, merged_summary_op])
            step += 1
            summary_writer.add_summary(summary, step)
            print 'step:',step,'mse:',mse,'time:',time.time()-s
            if step % 500 == 0:
               saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
               saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         pretrain = False
         print 'Done pretraining....training D and G now' 

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
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'
