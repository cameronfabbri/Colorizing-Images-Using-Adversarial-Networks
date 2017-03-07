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

   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(images_dir)
   except: pass
   
   global_step = tf.Variable(0, name='global_step', trainable=False)
  
   test_paths, trainData = data_ops.loadData(data_dir, dataset, batch_size)
   num_train = trainData.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = trainData.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = trainData.targets
   
   # this is testing stuff
   test_image = tf.placeholder(tf.float32, shape=(256, 256, 3), name='test_image')
  
   # test image in LAB color space
   test_image = data_ops.rgb_to_lab(test_image)
   
   # this is the test image in LAB with range [-1, 1]
   test_Lc, test_ac, test_bc = data_ops.preprocess_lab(test_image)
   test_L  = tf.expand_dims(test_Lc, axis=2)
   test_ab = tf.stack([test_ac, test_bc], axis=2)
   test_L  = tf.expand_dims(test_L, axis=0)
   test_ab = tf.expand_dims(test_ab, axis=0)
   
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
   
      #test_colored = colorarch.netG(test_L, batch_size)
  
   if loss_method == 'wasserstein':
      errD = tf.reduce_mean(errD_real - errD_fake)
      errG = tf.reduce_mean(errD_fake)

   if loss_method == 'energy':
      print 'using ebgans'



   #prediction = data_ops.augment(test_colored, test_L)
   #prediction = tf.image.convert_image_dtype(prediction, dtype=tf.uint8, saturate=True)
   ##############################################

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
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

   while True:
      epoch_num = step/(num_train/batch_size)
      s = time.time()
      
      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 100
      else: n_critic = 10

      # train the discriminator for 5 or 100 runs
      for critic_itr in range(n_critic):
         sess.run(D_train_op)
         sess.run(clip_discriminator_var_op)
      
      sess.run(G_train_op)

      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

      summary_writer.add_summary(summary, step)
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,' time:',time.time()-s
      step += 1
      
      if step%100 == 0:

         print 'Saving model...'
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n'
         
         '''
         print 'Evaluating...'
         random.shuffle(test_paths)
         test_paths_ = test_paths[:5]

         i = 0
         for t_image in test_paths_:
            img = misc.imread(t_image)
            img = misc.imresize(img, (256,256))
            colored = sess.run(prediction, feed_dict={test_image:img})

            test_L_image = sess.run(test_L, feed_dict={test_image:img})
            true_image   = np.uint8(sess.run(test_image, feed_dict={test_image:img}))
           
            pred_image = sess.run(prediction, feed_dict={test_image:img})[0]
            misc.imsave(images_dir+str(step)+'_'+str(i)+'_true.png', true_image)
            misc.imsave(images_dir+str(step)+'_'+str(i)+'_pred.png', pred_image)
            i += 1
         print 'Done evaluating....running the critic 100 times.'
         '''
