import tensorflow as tf
from architecture import netD, netG_encoder, netG_decoder
import numpy as np
import random
import ntpath
import sys
import cv2
import os

# for lab colorspace
from scipy import misc
from skimage import color

sys.path.insert(0, 'config/')
sys.path.insert(0, '../../ops/')
import loadceleba_colorize
from data_ops import normalizeImage, saveImage


'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):

   checkpoint_dir = info['checkpoint_dir']
   batch_size     = info['batch_size']
   dataset        = info['dataset']
   load           = info['load']

   # load data
   image_paths = loadceleba_colorize.load(load=load)
   color_train_data = np.asarray(image_paths['color_images_train'])
   gray_train_data  = np.asarray(image_paths['gray_images_train'])
   color_test_data  = np.asarray(image_paths['color_images_test'])
   gray_test_data   = np.asarray(image_paths['gray_images_test'])
   
   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   color_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='color_images')
   gray_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 1), name='gray_images')

   # images colorized by network
   #gen_images = netG(gray_images, batch_size)
   encoded_gen, conv7, conv6, conv5, conv4, conv3, conv2, conv1 = netG_encoder(gray_images, batch_size)
   decoded_gen = netG_decoder(encoded_gen, conv7, conv6, conv5, conv4, conv3, conv2, conv1, gray_images)
   
   # get the output from D on the real and fake data
   errD_real = netD(color_images, batch_size)
   errD_fake = netD(decoded_gen, batch_size, reuse=True) # gotta pass reuse=True to reuse weights

   # cost functions
   errD = tf.reduce_mean(errD_real - errD_fake)
   errG = tf.reduce_mean(errD_fake)

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   tf.summary.image('color_images', color_images, max_outputs=batch_size)
   tf.summary.image('generated_images', decoded_gen, max_outputs=batch_size)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # clip weights in D
   #clip_values = [-0.01, 0.01]
   clip_values = [-0.005, 0.005]
   clip_discriminator_var_op = [var.assign(tf.clip_by_value(var, clip_values[0], clip_values[1])) for
      var in d_vars]

   # optimize G
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errG, var_list=g_vars, global_step=global_step, colocate_gradients_with_ops=True)
   #G_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step, colocate_gradients_with_ops=True)
   #D_train_op = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(errD, var_list=d_vars, global_step=global_step)

   # change to use a fraction of memory
   #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
   init      = tf.global_variables_initializer()
   #sess      = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+dataset+'/logs/', graph=tf.get_default_graph())

   # only keep one model
   saver = tf.train.Saver(max_to_keep=1)
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir+dataset+'/')

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
   num_train_images = len(gray_train_data)
   num_test_images = len(gray_test_data)

   while True:

      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 100
      else: n_critic = 5

      # train the discriminator for 5 or 25 runs
      for critic_itr in range(n_critic):
         
         # have to load images from disk
         idx = np.random.choice(np.arange(num_train_images), batch_size, replace=False)

         # get color images
         batch_color_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
         batch_gray_images  = np.empty((batch_size, 256, 256, 1), dtype=np.float32)
         i = 0
         for cim, gim in zip(color_train_data[idx], gray_train_data[idx]):
            # read in color image
            cimg = misc.imread(cim)
            #cimg = color.rgb2lab(cimg)

            # read in gray image
            gimg = misc.imread(gim)
            gimg = np.expand_dims(gimg, 2)
            
            # now convert to float and put in tanh range
            cimg = normalizeImage(np.float32(cimg))
            gimg = normalizeImage(np.float32(gimg))

            batch_color_images[i, ...] = cimg
            batch_gray_images[i, ...]  = gimg

            i += 1

         sess.run(D_train_op, feed_dict={color_images:batch_color_images, gray_images:batch_gray_images})
         sess.run(clip_discriminator_var_op)

      idx = np.random.choice(np.arange(num_train_images), batch_size, replace=False)
      batch_gray_image = np.empty((batch_size, 256, 256, 1), dtype=np.float32)
      i = 0
      for gim in gray_train_data[idx]:
         gimg = misc.imread(gim)
         gimg = np.expand_dims(gimg, 2)
         gimg = normalizeImage(np.float32(gimg))
         batch_gray_images[i, ...] = gimg
         i += 1
      sess.run(G_train_op, feed_dict={gray_images:batch_gray_images})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                          feed_dict={color_images:batch_color_images,
                                          gray_images:batch_gray_images})

      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss

      step += 1

      #if step%1000 == 0:
      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+dataset+'/checkpoint-'+str(step), global_step=global_step)
         
         # evaluate on some test data
         print 'Evaluating...'
         idx = np.random.choice(np.arange(num_test_images), batch_size, replace=False)
        
         batch_color_images = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
         batch_gray_images  = np.empty((batch_size, 256, 256, 1), dtype=np.float32)

         i = 0
         for cim, gim in zip(color_test_data[idx], gray_test_data[idx]):
            cimg = misc.imread(cim)
            cimg = color.rgb2lab(cimg)
         
            gimg = misc.imread(gim)
            gimg = np.expand_dims(gimg, 2)
         
            cimg = normalizeImage(np.float32(cimg))
            gimg = normalizeImage(np.float32(gimg))
            
            batch_gray_images[i, ...]  = gimg
            batch_color_images[i, ...] = cimg
            i += 1

         gen_images = np.asarray(sess.run(decoded_gen, feed_dict={gray_images:batch_gray_images}))
         saveImage(gen_images, str(step), 'celeba')
         saveImage(gen_images, str(step), 'celeba')
