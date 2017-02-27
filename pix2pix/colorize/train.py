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
sys.path.insert(0, '../../ops/')
import loadceleba_colorize
from data_ops import normalizeImage, saveImage

'''
def read_data(filename_queue, batch_size):
   reader = tf.TFRecordReader()
   
   _, serialized_example = reader.read(filename_queue)
   
   features = tf.parse_single_example(serialized_example,
      features={
         'color_image': tf.FixedLenFeature([], tf.string),
         'gray_image':  tf.FixedLenFeature([], tf.string)
      })

   color_image = tf.decode_raw(features['color_image'], tf.float32)
   gray_image = tf.decode_raw(features['gray_image'], tf.float32)

   color_image_shape = tf.stack([256, 256, 3])
   gray_image_shape  = tf.stack([256, 256, 1])

   color_image = tf.reshape(color_image, color_image_shape)
   gray_image  = tf.reshape(gray_image, gray_image_shape)
   
   color_images, gray_images = tf.train.shuffle_batch(
      [color_image, gray_image],
      batch_size = batch_size,
      num_threads=12,
      capacity=10000+12*batch_size,
      min_after_dequeue=10000)

   return color_images, gray_images
'''

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):

   checkpoint_dir       = info['checkpoint_dir']
   batch_size           = info['batch_size']
   dataset              = info['dataset']
   load                 = info['load']

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)

   color_images = tf.placeholder(tf.float32, shape=(batch_size, 256, 256, 3), name='color_images')
   color_images = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), color_images)
   
   gray_images  = tf.map_fn(lambda img: tf.image.rgb_to_grayscale(img), color_images)
   gray_images  = tf.map_fn(lambda img: tf.image.random_flip_left_right(img), gray_images)

   # images colorized by network
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

   # optimize D
   D_train_op = tf.train.RMSPropOptimizer(learning_rate=0.00005).minimize(errD, var_list=d_vars, global_step=global_step, colocate_gradients_with_ops=True)

   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session()
   sess.run(init)

   #coord = tf.train.Coordinator()
   #threads = tf.train.start_queue_runners(sess=sess, coord=coord)
   
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
   #num_train_images = len(gray_train_data)
   #num_test_images = len(gray_test_data)
   print 'here'
   exit()
   try:
      while not coord.should_stop():
         s = time.time()
         # get the discriminator properly trained at the start
         if step < 25 or step % 500 == 0:
            n_critic = 100
         else: n_critic = 5

         # train the discriminator for 5 or 100 runs
         for critic_itr in range(n_critic):

            # need to read in a batch of images here
            feed_dict = getBatch(batch_size, dataset)


            sess.run([D_train_op, color_images, gray_images])
            sess.run(clip_discriminator_var_op)

         sess.run([G_train_op, gray_images])

         # now get all losses and summary *without* performing a training step - for tensorboard
         D_loss, G_loss, summary = sess.run([errD, errG, merged_summary_op])
         summary_writer.add_summary(summary, step)

         print 'step:',step,'D loss:',D_loss,'G_loss:',G_loss,' time:',time.time()-s

         # THIS WORKS TO RECOVER IMAGE
         #col_img = np.asarray(sess.run(color_images))[0]
         #col_img = (col_img+1)*127.5
         #col_img = color.lab2rgb(np.float64(col_img))
         #misc.imsave('test_im.jpg', col_img)

         step += 1

         if step%1 == 0:
            print 'Saving model...'
            saver.save(sess, checkpoint_dir+dataset+'/checkpoint-'+str(step), global_step=global_step)
            
            # evaluate on some test data
            print 'Evaluating...'
           
            gen_images = np.asarray(sess.run(decoded_gen))
            i = 0
            for img in gen_images:
               img = (img+1)*127.5
               img = color.lab2rgb(np.float64(img))
               misc.imsave('images/'+dataset+'/'+str(step)+'_'+str(i)+'.jpg', img)
               i += 1
               if i == 10: break

            exit()
            

            #saveImage(gen_images, str(step), 'celeba')
            #saveImage(gen_images, str(step), 'celeba')


   except tf.errors.OutOfRangeError:
      print 'Done training'
   finally:
      coord.request_stop()
   coord.join(threads)
   sess.close()
