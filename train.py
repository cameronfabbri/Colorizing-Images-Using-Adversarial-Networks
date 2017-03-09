import cPickle as pickle
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

   parser = argparse.ArgumentParser()
   parser.add_argument('--PRETRAIN_EPOCHS',required=True,type=int,help='Number of epochs to pretrain')
   parser.add_argument('--GAN_EPOCHS',     required=True,type=int,help='Number of epochs for GAN')
   parser.add_argument('--ARCHITECTURE',   required=True,help='Architecture for the generator')
   parser.add_argument('--DATASET',        required=True,help='The dataset to use')
   parser.add_argument('--DATA_DIR',       required=True,help='Directory where data is')
   parser.add_argument('--PRETRAIN_LR',    required=False,type=float,help='Learning rate for the pretrained network')
   parser.add_argument('--BATCH_SIZE',     required=False,type=int,default=32,help='Batch size to use')
   parser.add_argument('--GAN_LR',         required=False,type=float,default=2e-5,help='Learning rate for the GAN')
   parser.add_argument('--NUM_GPU',        required=False,type=int,default=1,help='Use multiple GPUs or not')
   parser.add_argument('--NUM_CRITIC',     required=False,type=int,default=10,help='Number of critics')
   parser.add_argument('--LOSS_METHOD',    required=False,default='wasserstein',help='Loss function for GAN',
      choices=['wasserstein','least_squares','energy'])
   parser.add_argument('--LOAD_MODEL', required=False,help='Load a trained model')
   a = parser.parse_args()

   PRETRAIN_EPOCHS = a.PRETRAIN_EPOCHS
   GAN_EPOCHS      = a.GAN_EPOCHS
   ARCHITECTURE    = a.ARCHITECTURE
   DATASET         = a.DATASET
   DATA_DIR        = a.DATA_DIR
   PRETRAIN_LR     = a.PRETRAIN_LR
   GAN_LR          = a.GAN_LR
   NUM_GPU         = a.NUM_GPU
   LOSS_METHOD     = a.LOSS_METHOD
   NUM_CRITIC      = a.NUM_CRITIC
   BATCH_SIZE      = a.BATCH_SIZE
   LOAD_MODEL      = a.LOAD_MODEL

   EXPERIMENT_DIR = 'checkpoints/'+ARCHITECTURE+'_'+DATASET+'_'+LOSS_METHOD+'_'+str(PRETRAIN_EPOCHS)+'_'+str(GAN_EPOCHS)+'_'+str(PRETRAIN_LR)+'_'+str(NUM_CRITIC)+'/'
   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(EXPERIMENT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['PRETRAIN_EPOCHS'] = PRETRAIN_EPOCHS
   exp_info['ARCHITECTURE']    = ARCHITECTURE
   exp_info['LOSS_METHOD']     = LOSS_METHOD
   exp_info['PRETRAIN_LR']     = PRETRAIN_LR
   exp_info['GAN_EPOCHS']      = GAN_EPOCHS
   exp_info['DATASET']         = DATASET
   exp_info['DATA_DIR']        = DATA_DIR
   exp_info['GAN_LR']          = GAN_LR
   exp_info['NUM_GPU']         = NUM_GPU
   exp_info['NUM_CRITIC']      = NUM_CRITIC
   exp_info['BATCH_SIZE']      = BATCH_SIZE
   exp_info['LOAD_MODEL']      = LOAD_MODEL
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'PRETRAIN_EPOCHS: ',PRETRAIN_EPOCHS
   print 'GAN_EPOCHS:      ',GAN_EPOCHS
   print 'ARCHITECTURE:    ',ARCHITECTURE
   print 'LOSS_METHOD:     ',LOSS_METHOD
   print 'PRETRAIN_LR:     ',PRETRAIN_LR
   print 'DATASET:         ',DATASET
   print 'DATA_DIR:        ',DATA_DIR
   print 'GAN_LR:          ',GAN_LR
   print 'NUM_GPU:         ',NUM_GPU
   print 'NUM_CRITIC:      ',NUM_CRITIC
   print 'LOAD_MODEL:      ',LOAD_MODEL
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE)
   # number of training images
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets

   # using the architecture from https://arxiv.org/pdf/1611.07004v1.pdf
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

   # architecture from
   # http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
   if ARCHITECTURE == 'colorarch':
      import colorarch
      # generate a colored image
      gen_img = colorarch.netG(L_image, BATCH_SIZE, NUM_GPU)

      # send real image to D
      errD_real = colorarch.netD(ab_image, BATCH_SIZE, NUM_GPU)

      # send generated image to D
      errD_fake = colorarch.netD(gen_img, BATCH_SIZE, NUM_GPU, reuse=True)
  
   if LOSS_METHOD == 'wasserstein':
      print 'Using Wasserstein loss'
      errD = tf.reduce_mean(errD_real - errD_fake)
      errG = tf.reduce_mean(errD_fake) + tf.reduce_mean((ab_image-gen_img)**2)

   if LOSS_METHOD == 'energy':
      print 'Using energy loss'
   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      errD = tf.reduce_mean(tf.square(errD_real - 1) + tf.square(errD_fake))
      errG = tf.reduce_mean(tf.square(errD_fake - 1))

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   if LOSS_METHOD == 'wasserstein':
      # clip weights in D
      clip_values = [-0.005, 0.005]
      clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # MSE loss for pretraining
   if PRETRAIN_EPOCHS > 0:
      print 'Pretraining generator for',PRETRAIN_EPOCHS,'epochs...'
      mse_loss = tf.reduce_mean((ab_image-gen_img)**2)
      mse_train_op = tf.train.AdamOptimizer(learning_rate=PRETRAIN_LR).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      tf.add_to_collection('vars', mse_train_op)
      tf.summary.scalar('mse_loss', mse_loss)

   if LOSS_METHOD == 'wasserstein':
      G_train_op = tf.train.RMSPropOptimizer(learning_rate=GAN_LR).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.RMSPropOptimizer(learning_rate=GAN_LR).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)

   else:
      G_train_op = tf.train.AdamOptimizer(learning_rate=GAN_LR).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.AdamOptimizer(learning_rate=GAN_LR).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())

   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   # only keep one model
   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   # restore previous model if there is one
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
 
   if LOAD_MODEL:
      ckpt = tf.train.get_checkpoint_state(LOAD_MODEL)
      print "Restoring model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         raise

   ########################################### training portion
   step = sess.run(global_step)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()
   start = time.time()
   while True:
      # if PRETRAIN, don't run G or D until number of epochs is met
      epoch_num = step/(num_train/BATCH_SIZE)
      
      while epoch_num < PRETRAIN_EPOCHS:
         epoch_num = step/(num_train/BATCH_SIZE)
         s = time.time()
         sess.run(mse_train_op)
         mse, summary = sess.run([mse_loss, merged_summary_op])
         step += 1
         summary_writer.add_summary(summary, step)
         print 'step:',step,'mse:',mse,'time:',time.time()-s
         if step % 500 == 0:
            print
            print 'Saving model'
            print
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      print
      print 'Done pretraing....training D and G now'
      print
      epoch_num = 0
      while epoch_num < GAN_EPOCHS:
         s = time.time()
         if LOSS_METHOD == 'wasserstein':
            if step < 25 or step % 500 == 0:
               n_critic = 50
            else: n_critic = 5

            for critic_itr in range(n_critic):
               sess.run(D_train_op)
               if LOSS_METHOD == 'wasserstein': sess.run(clip_discriminator_var_op)
           
            sess.run(G_train_op)
            D_loss, D_loss_f, D_loss_r, G_loss, summary = sess.run([errD, tf.reduce_mean(errD_fake), tf.reduce_mean(errD_real), errG, merged_summary_op])

         # For least squares it's 1:1 for D and G
         elif LOSS_METHOD == 'least_squares':
            sess.run(D_train_op)
            sess.run(G_train_op)
            D_loss, D_loss_f, D_loss_r, G_loss, summary = sess.run([errD, tf.reduce_mean(errD_fake), tf.reduce_mean(errD_real), errG, merged_summary_op])

         summary_writer.add_summary(summary, step)
         print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'D_loss_fake:',D_loss_f,'D_loss_r:',D_loss_r,'G_loss:',G_loss,' time:',time.time()-s
         step += 1
         
         if step%100 == 0:
            print 'Saving model...'
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
            print 'Model saved\n'

      print 'Finished training', time.time()-start
      saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
      saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      exit()
