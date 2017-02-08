'''

   Implementation of https://arxiv.org/abs/1511.06434

   Also imploring some of the tips from here:
   https://github.com/soumith/ganhacks

   Cameron Fabbri
   2/7/2017

   Going to be resizing images on the fly because I am not resizing all of imagenet
   just for this implementation when we are going to be using 224x224 for the main
   part of the project anyway.

   Training details from the paper

      - Data should be scaled to [-1, 1] for the TanH activation function.
      - Use Adam optimizer with learning rate of 0.0002, B1 = 0.5
      - Mini-batch size of 128
      - All weights are initialized from a zero-centered normal distribution with a stddev of 0.02
      - for leakyrelu, the slope was 0.2

   This can be trained with imagenet, lsun, or mnist

   Use soft and noisy labels: Label Smoothing, i.e. if you have two target labels: Real=1 and Fake=0,
   then for each incoming sample, if it is real, then replace the label with a random number between
   0.7 and 1.2, and if it is a fake sample, replace it with 0.0 and 0.3 (for example)

   Occassionally flip the labels when training the discriminator

   Use experience replay from RL, keep a replay buffer of past generations and occassionally show them.
   Same with G and D


'''

import tensorflow as tf
import cPickle as pickle
import sys
import random
import numpy as np
import cv2
import os

sys.path.insert(0, '../architecture/')
from architecture import generator, discriminator


def train(batch_size, checkpoint_dir, data):

   train_size = len(data)

   # placeholder to pass to the generator and descriminator to indicate training or not
   training = tf.placeholder(tf.bool, name='training')

   # create a step counter that will be saved out with the model
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # images from the true dataset
   images_d = tf.placeholder(tf.float32, shape=(batch_size, 64, 64, 3), name='images_d')

   # no need for gen images placeholder because they get generated below.

   # labels for the loss function since I will use label smoothing
   pos_labels = tf.placeholder(tf.float32, shape=(batch_size, 1), name='pos_label')
   neg_labels = tf.placeholder(tf.float32, shape=(batch_size, 1), name='neg_label')

   # placeholder for z, which is fed into the generator.
   z = tf.placeholder(tf.float32, shape=(batch_size, 100), name='z')

   # get a generated image from G
   generated_image = generator(z, batch_size)

   # send the real images to D
   D_real = discriminator(images_d, batch_size, train=training)

   # returns D's decision on the generated images
   D_gen  = discriminator(generated_image, batch_size, reuse=True, train=training)

   # compute the loss for D on the real images
   D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(pos_labels, D_real))
   
   # compute the loss for D on the generated images
   D_loss_gen  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(neg_labels, D_gen))

   # combine both losses for D
   D_loss = D_loss_real+D_loss_gen

   # G loss is to maximize log(D(G(z))), aka minimize the inverse
   G_loss = tf.reduce_mean(-tf.log(D_gen))

   # get the variables that can be trained, aka the layers in G and D (look at names)
   t_vars = tf.trainable_variables()

   # get the variables from both that we need to train
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # run the optimizer
   D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(D_loss, var_list=d_vars)
   G_train_op = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5).minimize(G_loss, var_list=g_vars)

   # initialize global variables, then create a session
   init      = tf.global_variables_initializer()
   sess      = tf.Session()

   saver = tf.train.Saver()

   try: os.mkdir(checkpoint_dir)
   except: pass

   # run the session with the variables
   sess.run(init)

   # check to see if there is a previous model. If so, load it.
   ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass

   # get the current step. If just starting then 0, else it will be loaded from the previous model.
   step = int(sess.run(global_step))
   epoch_num = step/(train_size/batch_size)

   # train forever
   while True:

      # sample from a normal distribution instead of a uniform distribution 
      batch_z = np.random.normal(-1, 1, [batch_size, 100]).astype(np.float32)
      
      # get random batch of image paths
      batch_paths = random.sample(data, batch_size)
     
      # create noisy positive and negative labels
      p_lab = np.random.uniform(0.7, 1.2, [batch_size, 1])
      n_lab = np.random.uniform(0.0, 0.3, [batch_size, 1])

      batch_real_images = []
      for img in batch_paths:
         img = cv2.imread(img).astype('float32') # read in image
         img = cv2.resize(img, (64,64)) # resize 
         img = 2*((img-np.min(img))/(np.max(img)-np.min(img))) - 1 # scale to [-1, 1]
         batch_real_images.append(img)

      batch_real_images = np.asarray(batch_real_images)
     
      _, d_loss_gen, d_loss_real, d_tot_loss = sess.run([D_train_op, D_loss_gen, D_loss_real, D_loss],
         feed_dict={images_d: batch_real_images, z: batch_z, pos_labels: p_lab, neg_labels: n_lab, training:True})

      _, g_loss, gen_images = sess.run([G_train_op, G_loss, generated_image], feed_dict={z:batch_z, training:True})

      print 'epoch:',epoch_num,'step:',step
      print 'd_loss:',d_tot_loss
      print 'g_loss:',g_loss
      print
      step += 1
      
      if step % 100 == 0:

         print 'Saving model'
         saver.save(sess, checkpoint_dir+'checkpoint_', global_step=global_step)
         print 'Evaluating...'
         _, g_loss, gen_images = sess.run([G_train_op, G_loss, generated_image], feed_dict={z:batch_z, training:False})

         random.shuffle(gen_images)

         count = 0
         for img in gen_images:
            cv2.imwrite('images/step_'+str(step)+'_'+str(count)+'.png', img)
            count += 1
            if count == 10: break



def main():

   dataset = 'imagenet'
   batch_size = 128

   if dataset == 'imagenet':
      print 'Loading imagenet...'
      pf = open('../../files/imagenet_complete.pkl', 'rb')
      data = pickle.load(pf)
      pf.close()

   checkpoint_dir = 'models/'

   train(batch_size, checkpoint_dir, data)

if __name__ == '__main__': main()
