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

sys.path.insert(0, 'config/')
sys.path.insert(0, '../ops/')

import data_ops
import load_data

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):

   checkpoint_dir       = info['checkpoint_dir']
   batch_size           = info['batch_size']
   dataset              = info['dataset']
   use_labels           = info['use_labels']

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)

   color_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='color_images')
   color_images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), color_images)
   
   gray_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 1), name='gray_images')
   gray_images  = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), gray_images)

   label_size = 0
   if dataset == 'imagenet' and use_labels is True:
      label_size = 1000

   labels_p = tf.placeholder(tf.float32, shape=(batch_size, label_size), name='labels')

   # images colorized by network
   encoded_gen, conv7, conv6, conv5, conv4, conv3, conv2, conv1 = netG_encoder(gray_images, labels_p, batch_size, use_labels)
   decoded_gen = netG_decoder(encoded_gen, conv7, conv6, conv5, conv4, conv3, conv2, conv1, gray_images)
   
   # get the output from D on the real and fake data
   errD_real = netD(color_images, labels_p, batch_size, use_labels)
   errD_fake = netD(decoded_gen, labels_p, batch_size, use_labels, reuse=True) # gotta pass reuse=True to reuse weights

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

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step, colocate_gradients_with_ops=True)

   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+dataset+'/logs/', graph=tf.get_default_graph())


   tf.add_to_collection('vars', G_train_op)
   tf.add_to_collection('vars', D_train_op)

   #meta_graph_def = tf.train.export_meta_graph(filename='my-model-1.meta')

   # only keep one model
   #ckpt = tf.train.get_checkpoint_state(checkpoint_dir+dataset+'/')

   # restore previous model if there is one
   #if ckpt and ckpt.model_checkpoint_path:
   #   print "Restoring previous model..."
   #   try:
   #      saver.restore(sess, ckpt.model_checkpoint_path)
   #      print "Model restored"
   #   except:
   #      print "Could not restore model"
   #      pass
   
   ########################################### training portion

   # get data for dataset we're using
   # train_data contains [image_paths, labels]
   print 'Loading train data...'
   train_data = load_data.load(dataset, use_labels, 'train')
   print 'Loading test data...'
   test_data  = load_data.load(dataset, use_labels, 'test')
   
   random.shuffle(train_data)
   random.shuffle(test_data)

   step = sess.run(global_step)
   num_train = len(train_data)
   num_test  = len(test_data)

   print num_train, 'training images'
   print num_test, 'test images'


   while True:
      epoch_num = step/(num_train/batch_size)
      s = time.time()
      # get the discriminator properly trained at the start
      if step < 25 or step % 500 == 0:
         n_critic = 10
      else: n_critic = 5

      # train the discriminator for 5 or 100 runs
      for critic_itr in range(n_critic):

         # need to read in a batch of images here
         if use_labels:
            batch_c_imgs, batch_g_imgs, batch_labels = data_ops.getBatch(batch_size, train_data, dataset, use_labels)
            sess.run([D_train_op, color_images, gray_images], feed_dict={color_images:batch_c_imgs, gray_images:batch_g_imgs, labels_p:batch_labels})
         else:
            batch_c_imgs, batch_g_imgs = data_ops.getBatch(batch_size, train_data, dataset, use_labels)
            sess.run([D_train_op, color_images, gray_images], feed_dict={color_images:batch_c_imgs, gray_images:batch_g_imgs})
         
         sess.run(clip_discriminator_var_op)
      
      if use_labels:
         sess.run([G_train_op, gray_images], feed_dict={gray_images:batch_g_imgs,labels_p:batch_labels})
      else:
         sess.run([G_train_op, gray_images], feed_dict={gray_images:batch_g_imgs})

      # now get all losses and summary *without* performing a training step - for tensorboard
      D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op], feed_dict={color_images:batch_c_imgs,gray_images:batch_g_imgs,labels_p:batch_labels})
      summary_writer.add_summary(summary, step)

      print 'epoch:',epoch_num,'step:',step,'D loss:',D_loss,'G_loss:',G_loss,' time:',time.time()-s
      
      # THIS WORKS TO RECOVER IMAGE
      #col_img = np.asarray(sess.run(color_images))[0]
      #col_img = (col_img+1)*127.5
      #col_img = color.lab2rgb(np.float64(col_img))
      #misc.imsave('test_im.jpg', col_img)

      step += 1

      if step%1 == 0:
         print 'Saving model...'
         #saver.save(sess, checkpoint_dir+dataset+'/checkpoint-'+str(step), global_step=global_step)
         saver.save(sess, 'my-model-1')
         saver.export_meta_graph('my-model-1.meta')
         exit()
         print 'Model saved\n' 
         
         print 'Evaluating...'
         # get test images from test split
         test_c_imgs, test_g_imgs, test_labels = data_ops.getBatch(batch_size, test_data, dataset, use_labels)

         gen_images = np.asarray(sess.run(decoded_gen, feed_dict={gray_images:test_g_imgs}))

         j = 0
         
         for gimg, rimg in zip(gen_images, test_c_imgs):
            gimg = (gimg+1.0)*127.5
            gimg = color.lab2rgb(np.float64(gimg))
            rimg = misc.imresize(rimg, (256,256))
            misc.imsave('images/'+dataset+'_'+str(use_labels)+'/'+str(step)+'_'+str(j)+'_gen.png', gimg)
            misc.imsave('images/'+dataset+'_'+str(use_labels)+'/'+str(step)+'_'+str(j)+'_real.png', rimg)
            j += 1
            if j == 10: break
         exit()
