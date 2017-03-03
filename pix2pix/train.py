import tensorflow as tf
from architecture import netD, netG_encoder, netG_decoder
import numpy as np
import random
import ntpath
import sys
import cv2
import os
import time

from scipy import misc
from skimage import color
# for lab colorspace
from scipy import misc
from skimage import color

sys.path.insert(0, '../ops/')

import data_ops
import config

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(checkpoint_dir):

   batch_size     = config.batch_size
   data_dir       = config.data_dir
   dataset        = config.dataset

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   z           = tf.placeholder(tf.float32, shape=(batch_size, 100), name='z')
   test_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 1), name='test_images')

   #data = data_ops.load_data(data_dir, dataset)
  
   #input_images  = data.inputs  # gray (L) images
   #target_images = data_ops.augment(data.targets, input_images) # color (a b) images
   #num_train     = data.count

   train_images_list = data_ops.load_data(data_dir, dataset)
   filename_queue    = tf.train.string_input_producer(train_images_list)
   real_images       = data_ops.read_input_queue(filename_queue)


   conv8, conv7, conv6, conv5, conv4, conv3, conv2, conv1 = netG_encoder(input_images)
   decoded = data_ops.augment(netG_decoder(conv8, conv7, conv6, conv5, conv4, conv3, conv2, conv1, input_images), input_images)

   # get the output from D on the real and fake data
   errD_real = netD(target_images)
   errD_fake = netD(decoded, reuse=True) # gotta pass reuse=True to reuse weights

   l1_weight = 100.0
   # cost functions
   genL1 = tf.reduce_mean(tf.abs(target_images-decoded))
   errD = tf.reduce_mean(errD_real - errD_fake)
   errG = tf.reduce_mean(errD_fake) + genL1*l1_weight

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   tf.summary.image('input_images', input_images, max_outputs=batch_size)
   tf.summary.image('generated_images', decoded, max_outputs=batch_size)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # clip weights in D
   clip_values = [-0.01, 0.01]
   #clip_values = [-0.005, 0.005]
   clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step, colocate_gradients_with_ops=True)

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
   while True:
      epoch_num = step/(num_train/batch_size)
      s = time.time()
      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 1
      else: n_critic = 5

      # train the discriminator for 5 or 100 runs
      #for critic_itr in range(n_critic):
      print 'running d'
      #sess.run(D_train_op)
      print 'clipping values'
      #sess.run(clip_discriminator_var_op)
      
      print 'running g'
      #sess.run(G_train_op)

      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

      summary_writer.add_summary(summary, step)
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,' time:',time.time()-s
      step += 1

      if step%1 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n' 
         
         print 'Evaluating...'
         shuffle(test_images)
         for t in test_images:
            print t
            exit()


if __name__ == '__main__':

   checkpoint_dir = config.checkpoint_dir
   learning_rate  = config.learning_rate
   batch_size     = config.batch_size
   data_dir       = config.data_dir
   dataset        = config.dataset
   if checkpoint_dir[-1] is not '/': checkpoint_dir+='/'
   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(checkpoint_dir+dataset)
   except: pass
   try: os.mkdir('images/')
   except: pass
   try: os.mkdir('images/'+dataset)
   except: pass
   
   checkpoint_dir = checkpoint_dir+dataset
   
   buildAndTrain(checkpoint_dir)


