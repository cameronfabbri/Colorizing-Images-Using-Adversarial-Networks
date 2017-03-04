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
   
   test_image  = tf.placeholder(tf.float32, shape=(256, 256, 3), name='test_image')
   test_image  = data_ops.rgb_to_lab(test_image)

   test_Lc, test_ac, test_bc = data_ops.preprocess_lab(test_image)
   test_L = tf.expand_dims(test_Lc, axis=2)
   test_ab = tf.stack([test_ac, test_bc], axis=2)

   test_L = tf.expand_dims(test_L, axis=0)
   test_ab = tf.expand_dims(test_ab, axis=0)


   train_paths, test_paths, trainData = data_ops.loadTrainData(data_dir, dataset)
   
   num_train = trainData.count
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = trainData.inputs
   # The color channels in [-1, 1] range
   ab_image  = trainData.targets

   encoded, conv7, conv6, conv5, conv4, conv3, conv2, conv1 = netG_encoder(L_image)

   # send encoded part to decoder - as well as other layers for skip connections
   decoded = netG_decoder(encoded, conv7, conv6, conv5, conv4, conv3, conv2, conv1)

   enc_test_images, tconv7, tconv6, tconv5, tconv4, tconv3, tconv2, tconv1 = netG_encoder(test_L)
   dec_test_images = netG_decoder(enc_test_images, tconv7, tconv6, tconv5, tconv4, tconv3, tconv2, tconv1)
   
   prediction = data_ops.lab_to_rgb(data_ops.augment(dec_test_images, test_L))

   '''
      So I don't get confused:
      The GRAY image is sent to the encoder, gets encoded, THEN gets decoded but
      with an EXTRA channel. This is then compared to the ACTUAL ab image via L1.
   '''

   # find L1 loss of decoded and original -> this loss is combined with D loss
   l1_loss = tf.reduce_mean(tf.abs(decoded-ab_image))

   # send the real ab image to the critic
   errD_real = netD(ab_image)

   # send the fake ab image to the critic
   errD_fake = netD(decoded, reuse=True) # gotta pass reuse=True to reuse weights

   # weight of how much the l1 loss takes into account 
   l1_weight = 10.0
   # total error for the critic
   errD = tf.reduce_mean(errD_real - errD_fake)
   # error for the generator, including the L1 loss
   #errG = tf.reduce_mean(errD_fake) + l1_loss*l1_weight
   errG = tf.reduce_mean(errD_fake)

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   #tf.summary.image('input_images', input_images, max_outputs=batch_size)
   #tf.summary.image('generated_images', decoded, max_outputs=batch_size)
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
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)

   while True:
      epoch_num = step/(num_train/batch_size)
      s = time.time()
      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 1
      else: n_critic = 5

      # train the discriminator for 5 or 100 runs
      for critic_itr in range(n_critic):
         sess.run(D_train_op)
         sess.run(clip_discriminator_var_op)
      
      sess.run(G_train_op)

      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])

      summary_writer.add_summary(summary, step)
      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,' time:',time.time()-s
      step += 1
      
      if step%1 == 0:

         print 'Saving model...'
         #saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         #saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')
         print 'Model saved\n' 
         
         print 'Evaluating...'
         random.shuffle(test_paths)
         
         img = misc.imread(test_paths[0])
         img = misc.imresize(img, (256,256))
         colored = sess.run(dec_test_images, feed_dict={test_image:img})

         test_L_image = sess.run(test_L, feed_dict={test_image:img})
         true_image = np.uint8(sess.run(test_image, feed_dict={test_image:img}))
        
         pred = np.uint8(sess.run(prediction, feed_dict={test_image:img}))[0]
         misc.imsave('true_image.png', true_image)
         misc.imsave('prediction.png', pred)

         exit()
         batch_test_img = np.empty((batch_size, 256, 256, 1), dtype=np.float32)
         batch_test_color = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
         i = 0
         for t in test_images_paths:
            img = misc.imread(t)
            height, width, channels = img.shape
            if height is not 256 or width is not 256:
               img = misc.imresize(img, (256, 256))
               height, width, channels = img.shape
            try:
               batch_test_color[i, ...] = img
               lab_img = color.rgb2lab(img)
               L_img = lab_img[:,:,0]
               L_img = np.expand_dims(L_img, 2)
               batch_test_img[i, ...] = L_img
               i += 1
            except:
               continue
            if i == batch_size: break

         test_colored = sess.run(tdecoded, feed_dict={test_images:batch_test_img})

         for cim, rim in zip(test_colored, batch_test_color):
            # convert from lab to rgb
            L = batch_test_img[i]
            cim = (cim+1.)*127.5
            cim = color.lab2rgb(np.float64(cim))
            misc.imsave(str(step)+'_'+'real.png', rim)
            misc.imsave(str(step)+'_'+'gen.png', rim)
            exit()
            
         
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


