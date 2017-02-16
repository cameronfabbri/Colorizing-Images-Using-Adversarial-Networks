'''

   Implementation of https://arxiv.org/abs/1511.06434

   Also imploring some of the tips from here:
   https://github.com/soumith/ganhacks

   Cameron Fabbri
   2/7/2017

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
import matplotlib.pyplot as plt
import ntpath

sys.path.insert(0, '../architecture/')
from architecture import generator, discriminator

sys.path.insert(0, '../../ops/')
from tf_ops import tanh_scale, tanh_descale

sys.path.insert(0, 'config/')
import mnist

'''
   Creates the tensorflow placeholders
'''
def setup_params(dataset, batch_size, checkpoint_dir):
   
   # placeholder to pass to the generator and descriminator to indicate training or not
   training = tf.placeholder(tf.bool, name='training')
   
   try: os.mkdir(checkpoint_dir)
   except: pass

   try: os.mkdir(checkpoint_dir+'/'+dataset)
   except: pass

   try: os.mkdir('images/')
   except: pass

   try: os.mkdir('images/'+dataset)
   except: pass
   
   # images from the true dataset
   if dataset == 'imagenet' or dataset == 'lsun':
      color_images = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3), name='color_images')
      bw_images = tf.placeholder(tf.float32, shape=(batch_size, 224, 224, 3), name='bw_images')

   # labels for the loss function since I will use label smoothing
   pos_labels = tf.placeholder(tf.float32, shape=(batch_size, 1), name='pos_label')
   neg_labels = tf.placeholder(tf.float32, shape=(batch_size, 1), name='neg_label')

   learning_rate = tf.placeholder(tf.float32, shape=(1), name='learning_rate')

   return color_images, bw_images, pos_labels, neg_labels, learning_rate, training

def train(batch_size, checkpoint_dir, data, dataset, train_size, placeholders):

   logs_path = checkpoint_dir+dataset+'/logs/'

   color_images  = placeholders[0]
   bw_images     = placeholders[1]
   pos_labels    = placeholders[2]
   neg_labels    = placeholders[3]
   learning_rate = placeholders[4]
   training      = placeholders[5]

   # create a step counter that will be saved out with the model
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # get a generated image from G
   generated_image = generator(bw_images, batch_size, dataset, train=training)

   # send the real images to D
   D_real, _ = discriminator(color_images, batch_size, train=training)

   # returns D's decision on the generated images
   D_gen, D_gen_grad  = discriminator(generated_image, batch_size, reuse=True, train=training)

   # compute the loss for D on the real images
   
   #D_loss_real = tf.contrib.losses.cross_entropy(logits, pos_labels)   
   #D_loss_real = tf.reduce_mean(tf.contrib.losses.log_loss(D_real, pos_labels))
   #D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=pos_labels))
   #D_loss_real = tf.losses.add_loss.sigmoid_cross_entropy(logits=D_real, multi_class_labels=pos_labels)
   #tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=pos_labels))
   # compute the loss for D on the generated images
   #D_loss_gen  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gen, labels=neg_labels))
   #D_loss_gen  = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_gen, labels=neg_labels))
   #D_loss_gen = tf.reduce_mean(tf.contrib.losses.log_loss(D_gen, neg_labels))
   # combine both losses for D

   D_loss = tf.reduce_mean(-tf.log(D_real) - tf.log(1 - D_gen))
   G_loss = tf.reduce_mean(-tf.log(D_gen))

   # G loss is to maximize log(D(G(z))), aka minimize the inverse
   #G_loss = tf.reduce_mean(-tf.log(D_gen))
   #G_loss = tf.reduce_mean(1 - tf.log(D_gen))
   #G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(D_gen_grad, tf.ones_like(D_gen_grad)))
   #G_loss = tf.reduce_mean(-tf.log(D_gen_grad))
   #G_loss = tf.reduce_mean(tf.contrib.losses.log_loss(D_gen_grad, pos_labels))
   # create tensorboard summaries for viewing loss visually
   tf.summary.scalar('d_loss', D_loss)
   #tf.summary.scalar('d_loss_real', D_loss_real)
   #tf.summary.scalar('d_loss_gen', D_loss_gen)
   tf.summary.scalar('g_loss', G_loss)
   tf.summary.image('real_images', color_images, max_outputs=10)
   tf.summary.image('generated_images', generated_image, max_outputs=10)
   merged_summary_op = tf.summary.merge_all()

   # get the variables that can be trained, aka the layers in G and D (look at names)
   t_vars = tf.trainable_variables()
   
   # get the variables from both that we need to train
   d_vars = [var for var in t_vars if 'd_' in var.name]
   g_vars = [var for var in t_vars if 'g_' in var.name]

   # run the optimizer
   D_train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(D_loss, var_list=d_vars, global_step=global_step)
   G_train_op = tf.train.AdamOptimizer(learning_rate=0.0002).minimize(G_loss, var_list=g_vars, global_step=global_step)
   
   # stop tensorflow from using all of the GPU memory
   gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=1)

   # initialize global variables, then create a session
   init      = tf.global_variables_initializer()
   sess      = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
   #sess      = tf.Session()

   saver = tf.train.Saver(max_to_keep=1)
   
   # run the session with the variables
   sess.run(init)

   # write the summaries to tensorboard
   summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

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
   p_lab = np.ones([batch_size, 1])
   n_lab = np.zeros([batch_size, 1])

   color_paths = data['color_train']
   bw_paths    = data['gray_train']
   # train forever
   while True:

      # sample from a normal distribution instead of a uniform distribution 
      #batch_z = np.random.normal(-1, 1, [batch_size, 100]).astype(np.float32)
      #batch_z = np.random.uniform(-1, 1, [batch_size, 100]).astype(np.float32)
      
      # create noisy positive and negative labels
      #p_lab = np.random.uniform(0.7, 1.2, [batch_size, 1])
      #n_lab = np.random.uniform(0.0, 0.3, [batch_size, 1])

      #p_lab = np.ones(batch_size)
      #n_lab = np.zeros(batch_size)
      #p_lab = np.expand_dims(p_lab, 1)
      #n_lab = np.expand_dims(n_lab, 1)

      # get random batch of image paths if using imagenet or lsun
      if dataset == 'imagenet' or dataset == 'lsun':
         batch_color_images = np.empty((batch_size, 224, 224, 3), dtype=np.float32)
         batch_gray_images = np.empty((batch_size, 224, 224, 3), dtype=np.float32)
         
         bcolor_paths = random.sample(color_paths, batch_size)
         bbw_paths    = random.sample(bw_paths, batch_size)

         # get gray and color images
         i = 0
         for color,gray in zip(bcolor_paths, bbw_paths):
            color_img = cv2.imread(color).astype('float32') # read in image
            gray_img  = cv2.imread(gray).astype('float32') # read in image
            
            # scale to [-1, 1] for tanh
            #color_img = tanh_scale(color_img)
            #gray_img = tanh_scale(gray_img)
            
            batch_color_images[i, ...] = color_img
            batch_gray_images[i, ...]  = gray_img
            i += 1

      _, d_loss, summary = sess.run([D_train_op, D_loss, merged_summary_op], feed_dict={color_images: batch_color_images, bw_images: batch_gray_images, pos_labels: p_lab, neg_labels: n_lab, training:True})

      _, g_loss, gen_images = sess.run([G_train_op, G_loss, generated_image], feed_dict={bw_images:batch_gray_images, training:True})
      summary_writer.add_summary(summary, step)


      print 'epoch:',epoch_num,'step:',step
      print 'd_loss:',d_loss
      print 'g_loss:',g_loss
      print
      step += 1
      
      if step % 100 == 0:
      

         print 'Saving model...'
         saver.save(sess, checkpoint_dir+dataset+'/checkpoint-'+str(step), global_step=global_step)
         print 'Model saved\n'
         print 'Evaluating...'
         _, g_loss, gen_images = sess.run([G_train_op, G_loss, generated_image], feed_dict={bw_images:batch_gray_images, training:False})

         #random.shuffle(gen_images)

         count = 0
         for (gen, real) in zip(gen_images, batch_color_images):
            cv2.imwrite('images/'+dataset+'/step_'+str(step)+'_gen_'+str(count)+'.png', gen)
            cv2.imwrite('images/'+dataset+'/step_'+str(step)+'_real_'+str(count)+'.png', real)
            count += 1
            if count == 10: break


'''

   The main just loads up the parameters from the config file,
   creates tensorflow variables and stuff in setup_params, then
   calls train.

'''
def main():

   try:
      config_file = ntpath.basename(sys.argv[1]).split('.py')[0]
      config = __import__(config_file)
   except:
      print 'config',sys.argv[1],'not found'
      raise
      exit()

   # load parameters from config file passed in
   dataset        = config.dataset
   batch_size     = config.batch_size
   checkpoint_dir = config.checkpoint_dir
   learning_rate  = config.learning_rate
   dataset_location = config.dataset_location

   if dataset == 'imagenet':
      print 'Loading imagenet...'
      #pf = open('../../files/imagenet_bw.pkl', 'rb')
      pf = open(dataset_location, 'rb')
      data = pickle.load(pf)
      pf.close()

   if dataset == 'mnist':
      print 'Loading mnist...'

      # loads mnist and resizes to (28, 28, 1) as well as converts to float.
      train_images, val_images, test_images = mnist.load_mnist()

      # since we don't care about train/test/val, just group them together
      data = np.concatenate((train_images, val_images), axis=0)
      data = np.concatenate((data, test_images), axis=0)
  
   train_size = len(data['color_train'])

   # set up tensorflow variables to pass to train.
   color_images, bw_images, pos_labels, neg_labels, learning_rate, training = setup_params(dataset, batch_size, checkpoint_dir)
  
   placeholders    = []
   placeholders.append(color_images)
   placeholders.append(bw_images)
   placeholders.append(pos_labels)
   placeholders.append(neg_labels)
   placeholders.append(learning_rate)
   placeholders.append(training)
   
   train(batch_size, checkpoint_dir, data, dataset, train_size, placeholders)

if __name__ == '__main__': main()
