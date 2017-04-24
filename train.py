import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import pix2pix
import sys
import os
import time
import ColColGAN

sys.path.insert(0, 'ops/')
sys.path.insert(0, 'config/')
from tf_ops import *

import data_ops

if __name__ == '__main__':
   parser = argparse.ArgumentParser()
   parser.add_argument('--PRETRAIN_EPOCHS',required=False,default=0,type=int,help='Number of epochs to pretrain')
   parser.add_argument('--GAN_EPOCHS',     required=True,type=int,help='Number of epochs for GAN')
   parser.add_argument('--ARCHITECTURE',   required=True,help='Architecture for the generator')
   parser.add_argument('--DATASET',        required=True,help='The dataset to use')
   parser.add_argument('--DATA_DIR',       required=True,help='Directory where data is')
   parser.add_argument('--PRETRAIN_LR',    required=False,type=float,help='Learning rate for the pretrained network')
   parser.add_argument('--BATCH_SIZE',     required=False,type=int,default=32,help='Batch size to use')
   parser.add_argument('--GAN_LR',         required=False,type=float,default=2e-5,help='Learning rate for the GAN')
   parser.add_argument('--NUM_GPU',        required=False,type=int,default=1,help='Use multiple GPUs or not')
   parser.add_argument('--JITTER',         required=False,type=int,default=1,help='Whether or not to add jitter')
   parser.add_argument('--NUM_CRITIC',     required=False,type=int,default=5,help='Number of critics')
   parser.add_argument('--LOSS_METHOD',    required=False,default='wasserstein',help='Loss function for GAN',
      choices=['wasserstein','least_squares','energy','gan','cnn'])
   parser.add_argument('--SIZE',           required=False,default=256,help='size of the image',type=int)
   parser.add_argument('--LOAD_MODEL',     required=False,help='Load a trained model')
   parser.add_argument('--L1_WEIGHT',      required=False,help='weight of L1 for combined loss',type=float,default=100.0)
   parser.add_argument('--L2_WEIGHT',      required=False,help='weight of L2 for combined loss',type=float,default=0.0)
   parser.add_argument('--GAN_WEIGHT',     required=False,help='weight of GAN for combined loss',type=float,default=1.0)
   parser.add_argument('--UPCONVS',        required=False,help='flag for using upconvs or not',type=int,default=1)
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
   JITTER          = bool(a.JITTER)
   SIZE            = a.SIZE
   L1_WEIGHT       = a.L1_WEIGHT
   L2_WEIGHT       = a.L2_WEIGHT
   GAN_WEIGHT      = a.GAN_WEIGHT
   UPCONVS         = a.UPCONVS
   
   EXPERIMENT_DIR = 'checkpoints/'+ARCHITECTURE+'_'+DATASET+'_'+LOSS_METHOD+'_'+str(PRETRAIN_EPOCHS)+'_'+str(GAN_EPOCHS)+'_'+str(PRETRAIN_LR)+'_'+str(NUM_CRITIC)+'_'+str(GAN_LR)+'_'+str(JITTER)+'_'+str(SIZE)+'_'+str(L1_WEIGHT)+'_'+str(L2_WEIGHT)+'_'+str(GAN_WEIGHT)+'_'+str(UPCONVS)+'/'
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
   exp_info['JITTER']          = JITTER
   exp_info['SIZE']            = SIZE
   exp_info['L1_WEIGHT']       = L1_WEIGHT
   exp_info['L2_WEIGHT']       = L2_WEIGHT
   exp_info['GAN_WEIGHT']      = GAN_WEIGHT
   exp_info['UPCONVS']          = UPCONVS
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
   print 'JITTER:          ',JITTER
   print 'SIZE:            ',SIZE
   print 'L1_WEIGHT:       ',L1_WEIGHT
   print 'L2_WEIGHT:       ',L2_WEIGHT
   print 'GAN_WEIGHT:      ',GAN_WEIGHT
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE, jitter=JITTER, SIZE=SIZE)
   # number of training images
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets

   if ARCHITECTURE == 'pix2pix':
      # generated ab values from generator
      gen_ab = pix2pix.netG(L_image, NUM_GPU, UPCONVS)
   elif ARCHITECTURE == 'colcolgan':
      gen_ab = ColColGAN.netG(L_image, NUM_GPU)

   # D's decision on real images and fake images
   if LOSS_METHOD == 'energy':
      D_real, embeddings_real, decoded_real = pix2pix.energyNetD(L_image, ab_image, BATCH_SIZE)
      D_fake, embeddings_fake, decoded_fake = pix2pix.energyNetD(L_image, gen_ab, BATCH_SIZE, reuse=True)
   else:
      if ARCHITECTURE == 'pix2pix':
         D_real = pix2pix.netD(L_image, ab_image, NUM_GPU)
         D_fake = pix2pix.netD(L_image, gen_ab, NUM_GPU, reuse=True)
      elif ARCHITECTURE == 'colcolgan':
         D_real = ColColGAN.netD(L_image, ab_image, NUM_GPU)
         D_fake = ColColGAN.netD(L_image, gen_ab, NUM_GPU, reuse=True)

   e = 1e-12
   if LOSS_METHOD == 'wasserstein':
      print 'Using Wasserstein loss'
      D_real = lrelu(D_real)
      D_fake = lrelu(D_fake)
      gen_loss_GAN = tf.reduce_mean(D_fake)
      
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         errG = gen_loss_GAN
      errD = tf.reduce_mean(D_real - D_fake)

   if LOSS_METHOD == 'least_squares':
      print 'Using least squares loss'
      # Least squares requires sigmoid activation on D
      errD_real = tf.nn.sigmoid(D_real)
      errD_fake = tf.nn.sigmoid(D_fake)
      
      #gen_loss_GAN = tf.reduce_mean(tf.square(errD_fake - 1))
      gen_loss_GAN = 0.5*(tf.reduce_mean(tf.square(errD_fake - 1)))
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      elif L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         errG = gen_loss_GAN
      #errD = tf.reduce_mean(tf.square(errD_real - 1) + tf.square(errD_fake))
      errD = tf.reduce_mean(0.5*(tf.square(errD_real - 1)) + 0.5*(tf.square(errD_fake)))

   if LOSS_METHOD == 'gan' or LOSS_METHOD == 'cnn':
      print 'Using original GAN loss'
      if LOSS_METHOD is not 'cnn':
         D_real = tf.nn.sigmoid(D_real)
         D_fake = tf.nn.sigmoid(D_fake)
         gen_loss_GAN = tf.reduce_mean(-tf.log(D_fake + e))
      else: gen_loss_GAN = 0.0
      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1  = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2  = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG         = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using GAN loss, no L1 or L2'
         #errD = errD + e
         errG = gen_loss_GAN
      errD = tf.reduce_mean(-(tf.log(D_real+e)+tf.log(1-D_fake+e)))
   
   if LOSS_METHOD == 'energy':
      print 'Using energy loss'
      margin = 80
      gen_loss_GAN = D_fake

      zero = tf.zeros_like(margin-D_fake)

      if L1_WEIGHT > 0.0:
         print 'Using an L1 weight of',L1_WEIGHT
         gen_loss_L1 = tf.reduce_mean(tf.abs(ab_image-gen_ab))
         errG        = gen_loss_GAN*GAN_WEIGHT + gen_loss_L1*L1_WEIGHT
      if L2_WEIGHT > 0.0:
         print 'Using an L2 weight of',L2_WEIGHT
         gen_loss_L2 = tf.reduce_mean(tf.nn.l2_loss(ab_image-gen_ab))
         errG        = gen_loss_GAN*GAN_WEIGHT + gen_loss_L2*L2_WEIGHT
      if L1_WEIGHT <= 0.0 and L2_WEIGHT <= 0.0:
         print 'Just using energy loss, no L1 or L2'
         errG = gen_loss_GAN
      errD = D_real + tf.maximum(zero, margin-D_fake)

   # tensorboard summaries
   try: tf.summary.scalar('d_loss', tf.reduce_mean(errD))
   except:pass
   try: tf.summary.scalar('g_loss', tf.reduce_mean(errG))
   except:pass

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   if LOSS_METHOD == 'wasserstein':
      # clip weights in D
      clip_values = [-0.005, 0.005]
      clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for var in d_vars]

   # MSE loss for pretraining
   if PRETRAIN_EPOCHS > 0:
      print 'Pretraining generator for',PRETRAIN_EPOCHS,'epochs...'
      mse_loss = tf.reduce_mean((ab_image-gen_ab)**2)
      #mse_train_op = tf.train.AdamOptimizer(learning_rate=PRETRAIN_LR,beta1=0.5).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      mse_train_op = tf.train.AdamOptimizer(learning_rate=PRETRAIN_LR).minimize(mse_loss, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      tf.add_to_collection('vars', mse_train_op)
      tf.summary.scalar('mse_loss', mse_loss)
   if LOSS_METHOD == 'wasserstein':
      G_train_op = tf.train.RMSPropOptimizer(learning_rate=GAN_LR, decay=0.9).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      D_train_op = tf.train.RMSPropOptimizer(learning_rate=GAN_LR, decay=0.9).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)
   else:
      #G_train_op = tf.train.AdamOptimizer(learning_rate=GAN_LR,beta1=0.5).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
      #D_train_op = tf.train.AdamOptimizer(learning_rate=GAN_LR,beta1=0.5).minimize(errD, var_list=d_vars, colocate_gradients_with_ops=True)
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
      if PRETRAIN_EPOCHS > 0:
         print 'Done pretraining....training D and G now'
         epoch_num = 0
      while epoch_num < GAN_EPOCHS:
         epoch_num = step/(num_train/BATCH_SIZE)
         s = time.time()

         if LOSS_METHOD == 'wasserstein':
            if step < 10 or step % 500 == 0:
               n_critic = 100
            else: n_critic = NUM_CRITIC
            for critic_itr in range(n_critic):
               sess.run(D_train_op)
               sess.run(clip_discriminator_var_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         
         elif LOSS_METHOD == 'least_squares':
            sess.run(D_train_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         elif LOSS_METHOD == 'gan'or LOSS_METHOD == 'energy':
            sess.run(D_train_op)
            sess.run(G_train_op)
            D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         elif LOSS_METHOD == 'cnn':
            sess.run(G_train_op)
            loss, summary = sess.run([errG, merged_summary_op])

         summary_writer.add_summary(summary, step)
         if LOSS_METHOD != 'cnn' and step%10==0: print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,'time:',time.time()-s
         else:
            if step%50==0:print 'epoch:',epoch_num,'step:',step,'loss:',loss,' time:',time.time()-s
         step += 1
         
         if step%500 == 0:
            print 'Saving model...'
            saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
            saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
            print 'Model saved\n'

      print 'Finished training', time.time()-start
      saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
      saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
      exit()
