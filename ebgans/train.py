import tensorflow as tf
from architecture import encoder, decoder, netD, netG, mse
import numpy as np
import random
import ntpath
import sys
import cv2
import os

sys.path.insert(0, '../ops/')

import loadceleba

'''
   Builds the graph and sets up params, then starts training
'''
def buildAndTrain(info):

   checkpoint_dir = info['checkpoint_dir']
   batch_size     = info['batch_size']
   dataset        = info['dataset']
   data_dir       = info['data_dir']
   use_pt         = info['use_pt']
   load           = info['load']

   # load data
   image_data = loadceleba.load(load=load, data_dir=data_dir)

   # placeholders for data going into the network
   global_step = tf.Variable(0, name='global_step', trainable=False)
   real_images = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name='color_images')
   z           = tf.placeholder(tf.float32, shape=(batch_size, 100), name='z')

   # generated images
   gen_images = netG(z, batch_size)

   D_loss_real, encoded_real, decoded_real = netD(real_images, batch_size)
   D_loss_fake, encoded_fake, decoded_fake = netD(gen_images, batch_size, reuse=True)

   # margin for celeba
   margin = 20

   # cost functions
   errD = D_loss_real + margin-D_loss_fake
   errG = D_loss_fake

   # tensorboard summaries
   tf.summary.scalar('d_loss', errD)
   tf.summary.scalar('g_loss', errG)
   tf.summary.image('real_images', real_images, max_outputs=batch_size)
   tf.summary.image('generated_images', gen_images, max_outputs=batch_size)
   merged_summary_op = tf.summary.merge_all()

   # get all trainable variables, and split by network G and network D
   t_vars = tf.trainable_variables()
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # optimize G
   G_train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(errG, var_list=g_vars, global_step=global_step)

   # optimize D
   D_train_op = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(errD, var_list=d_vars, global_step=global_step)

   saver = tf.train.Saver(max_to_keep=1)
   init  = tf.global_variables_initializer()
   
   sess  = tf.Session()
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(checkpoint_dir+'/'+'logs/', graph=tf.get_default_graph())

   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)

   tf.add_to_collection('G_train_op', G_train_op)
   tf.add_to_collection('D_train_op', D_train_op)

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

      batch_real_images = random.sample(image_data, batch_size)
      batch_z = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)

      sess.run(D_train_op, feed_dict={real_images:batch_real_images, z:batch_z})
      sess.run(G_train_op, feed_dict={real_images:batch_real_images, z:batch_z})
      sess.run(G_train_op, feed_dict={real_images:batch_real_images, z:batch_z})

      Gerr, Derr, summary = sess.run([errG, errD, merged_summary_op], feed_dict={real_images:batch_real_images, z:batch_z})

      summary_writer.add_summary(summary, step)

      print 'step:',step,'D loss:',Derr,'G_loss:',Gerr
      step += 1

      if step%500 == 0:
         print 'Saving model...'
         saver.save(sess, checkpoint_dir+'checkpoint-'+str(step))
         saver.export_meta_graph(checkpoint_dir+'checkpoint-'+str(step)+'.meta')

         batch_z  = np.random.uniform(-1.0, 1.0, size=[batch_size, 100]).astype(np.float32)
         gen_imgs = sess.run([gen_images], feed_dict={z:batch_z})

         num = 0
         for img in gen_imgs[0]:
            img = np.asarray(img)
            img = (img+1.)/2. # these two lines properly scale from [-1, 1] to [0, 255]
            img *= 255.0/img.max()
            cv2.imwrite('images/'+dataset+'_'+str(use_pt)+'_'+str(step)+'_'+str(num)+'.png', img)
            num += 1
            if num == 5:
               break
         print 'Done saving'


