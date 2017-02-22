import tensorflow as tf
from architecture import netD, netG
import numpy as np
import random
import ntpath
import sys
import cv2
import os

sys.path.insert(0, 'config/')
sys.path.insert(0, '../../ops/')
import loadceleba

from tf_ops import tanh_scale, tanh_descale

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):

   checkpoint_dir = info['checkpoint_dir']
   batch_size     = info['batch_size']
   dataset        = info['dataset']
   load           = info['load']

   # load data
   color_image_data, gray_image_data = loadceleba.load(load=load)

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   color_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='color_images')
   gray_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 1), name='gray_images')

   # images colorized by network
   gen_images = netG(gray_images, batch_size)

   # get the output from D on the real and fake data
   errD_real = netD(real_images, batch_size)
   errD_fake = netD(gen_images, batch_size, reuse=True) # gotta pass reuse=True to reuse weights

   # cost functions
   errD = tf.reduce_mean(errD_real - errD_fake)
   errG = tf.reduce_mean(errD_fake)

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   tf.summary.image('color_images', color_images, max_outputs=batch_size)
   tf.summary.image('generated_images', gen_images, max_outputs=batch_size)
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
   G_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step)

   # change to use a fraction of memory
   #gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)
   init      = tf.global_variables_initializer()
   #sess      = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+dataset+'/'+'logs/', graph=tf.get_default_graph())

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
   num_images = len(gray_image_data)

   while True:

      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 100
      else: n_critic = 5

      # train the discriminator for 5 or 25 runs
      for critic_itr in range(n_critic):
         #batch_real_images = random.sample(image_data, batch_size)
         #batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)

         # have to load images from disk
         idx = np.random.choice(np.arange(num_images), batch_size, replace=False)

         # get color images
         batch_color_images = np.empty((num_images, 256, 256, 3), dtype=np.float32)
         batch_gray_image   = np.empty((num_images, 256, 256, 1), dtype=np.float32)

         for cim, gim in zip(color_image_data[idx], gray_image_data[idx]):
            batch_color_images[i, ...] = tanh_scale(cv2.imread(cim).astype('float32'))
            batch_gray_images[i, ...]  = tanh_scale(cv2.imread(gim).astype('float32'))
         sess.run(D_train_op, feed_dict={color_images:batch_color_images, gray_images:batch_gray_images})
         sess.run(clip_discriminator_var_op)

      idx = np.random.choice(np.arange(num_images), batch_size, replace=False)
      batch_gray_image = np.empty((num_images, 256, 256, 1), dtype=np.float32)
      for gim in gray_image_data[idx]:
         batch_gray_images[i, ...] = tanh_scale(cv2.imread(gim).astype('float32'))
      sess.run(G_train_op, feed_dict={gray_images:batch_gray_images})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op],
                                          feed_dict={real_images:batch_real_images, z:batch_z})

      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss

      step += 1

      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+dataset+'/checkpoint-'+str(step), global_step=global_step)
         
         batch_z  = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         num = 0
         for img in gen_imgs[0]:
            img = np.asarray(img)
            #img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
            #img *= 255.0/img.max()
            img = tanh_descale(img)
            cv2.imwrite('images/'+dataset+'/'+str(step)+'_'+str(num)+'.png', img)
            num += 1
            if num == 20:
               break
         print 'Done saving'







