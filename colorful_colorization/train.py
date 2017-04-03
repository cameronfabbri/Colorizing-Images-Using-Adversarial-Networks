###
# python train.py --ARCHITECTURE 'basic' --LOSS_METHOD 'L1L2'
# python train.py --ARCHITECTURE 'ColCol' --LOSS_METHOD 'L2'
###

import cPickle as pickle
import tensorflow as tf
from scipy import misc
import numpy as np
import argparse
import ntpath
import sys
import os
import time

sys.path.insert(0, '../ops/')
sys.path.insert(0, 'config/')

import data_ops
import CONFIG_
import basic_ARCH
import ColCol_ARCH


if __name__ == '__main__':

   parser = argparse.ArgumentParser()
   parser.add_argument('--ARCHITECTURE',   required=True,help='Architecture for the model')
   parser.add_argument('--LOSS_METHOD',    required=False,default='MSE',help='Loss function for G',
      choices=['wasserstein','least_squares','L1L2', 'L2', 'L1'])
   
   a = parser.parse_args()

   ARCHITECTURE    = a.ARCHITECTURE
   LOSS_METHOD     = a.LOSS_METHOD
   PRETRAIN_EPOCHS = CONFIG_.PRETRAIN_EPOCHS
   DATASET         = CONFIG_.DATASET
   BATCH_SIZE      = CONFIG_.BATCH_SIZE
   PRETRAIN_LR     = CONFIG_.PRETRAIN_LR
   L_rate          = CONFIG_.L_rate
   NUM_GPU         = CONFIG_.NUM_GPU
   DATA_DIR        = CONFIG_.DATA_DIR
   LOAD_MODEL      = CONFIG_.LOAD_MODEL
   IMAGE_SIZE_     = 256

   EXPERIMENT_DIR = 'checkpoints/'+ARCHITECTURE+'_'+DATASET+'_'+LOSS_METHOD+'/'
   IMAGES_DIR = EXPERIMENT_DIR+'images/'

   print
   print 'Creating',EXPERIMENT_DIR
   try: os.mkdir('checkpoints/')
   except: pass
   try: os.mkdir(EXPERIMENT_DIR)
   except: pass
   try: os.mkdir(IMAGES_DIR)
   except: pass
   
   # write all this info to a pickle file in the experiments directory
   exp_info = dict()
   exp_info['PRETRAIN_EPOCHS'] = PRETRAIN_EPOCHS
   exp_info['ARCHITECTURE']    = ARCHITECTURE
   exp_info['LOSS_METHOD']     = LOSS_METHOD
   exp_info['PRETRAIN_LR']     = PRETRAIN_LR
   exp_info['L_rate']          = L_rate
   exp_info['DATASET']         = DATASET
   exp_info['DATA_DIR']        = DATA_DIR
   exp_info['NUM_GPU']         = NUM_GPU
   exp_info['BATCH_SIZE']      = BATCH_SIZE
   exp_pkl = open(EXPERIMENT_DIR+'info.pkl', 'wb')
   data = pickle.dumps(exp_info)
   exp_pkl.write(data)
   exp_pkl.close()
   
   print
   print 'PRETRAIN_EPOCHS: ',PRETRAIN_EPOCHS
   print 'ARCHITECTURE:    ',ARCHITECTURE
   print 'LOSS_METHOD:     ',LOSS_METHOD
   print 'PRETRAIN_LR:     ',PRETRAIN_LR
   print 'DATASET:         ',DATASET
   print 'DATA_DIR:        ',DATA_DIR
   print 'NUM_GPU:         ',NUM_GPU
   print

   # global step that is saved with a model to keep track of how many steps/epochs
   global_step = tf.Variable(0, name='global_step', trainable=False)

   # load data
   Data = data_ops.loadData(DATA_DIR, DATASET, BATCH_SIZE)
   
   # number of training images
   num_train = Data.count
   
   # The gray 'lightness' channel in range [-1, 1]
   L_image   = Data.inputs
   
   # The color channels in [-1, 1] range
   ab_image  = Data.targets

   # using the architecture from https://arxiv.org/pdf/1611.07004v1.pdf
   #if ARCHITECTURE == 'pix2pix':
      
   # architecture from
   # http://hi.cs.waseda.ac.jp/~iizuka/projects/colorization/data/colorization_sig2016.pdf
   #if ARCHITECTURE == 'colorarch':
 
   if ARCHITECTURE == 'basic':
         print 'Basic Architecture'
         gen_img = basic_ARCH.netG(L_image, BATCH_SIZE, NUM_GPU)
   if ARCHITECTURE == 'ColCol':
         print 'Basic Architecture'
         gen_img = ColCol_ARCH.netG(L_image, NUM_GPU)
    
   
   pred = data_ops.augment(gen_img, L_image)
   trueI = data_ops.augment(ab_image, L_image)
 
      
      
   if LOSS_METHOD == 'L2':
         print 'Using L2 loss'
         #gen_loss_mse = tf.reduce_mean(2 * tf.nn.l2_loss(gen_img - ab_image)) / (IMAGE_SIZE_ * IMAGE_SIZE_ * 100 * 100)
	 #gen_loss = tf.reduce_mean(tf.pow(trueI-pred, 2))
         gen_loss = tf.reduce_mean(tf.nn.l2_loss(ab_image - gen_img))
         tf.summary.scalar('L2', gen_loss)
         tf.summary.image('gen. images', pred, max_outputs=8)
      
   if LOSS_METHOD == 'L1':
         # Least squares requires sigmoid activation on D    
         gen_loss  = tf.reduce_mean(tf.abs(ab_image-gen_img) * (IMAGE_SIZE_ * IMAGE_SIZE_) )
         tf.summary.scalar('L1', gen_loss)
         tf.summary.image('gen. images', pred, max_outputs=8)
      
   
   t_vars = tf.trainable_variables()
   
   train_op = tf.train.AdamOptimizer(learning_rate=L_rate).minimize(gen_loss, var_list=t_vars, 
			global_step=global_step, colocate_gradients_with_ops=True)
   tf.add_to_collection('vars', train_op)
   
   
   ######################################################################## training
   saver = tf.train.Saver(max_to_keep=1)
   
   init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
   sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
   sess.run(init)

   # write out logs for tensorboard to the checkpointSdir
   summary_writer = tf.summary.FileWriter(EXPERIMENT_DIR+'/logs/', graph=tf.get_default_graph())
   tf.add_to_collection('vars', train_op)
   
   # only keep one model
   ckpt = tf.train.get_checkpoint_state(EXPERIMENT_DIR)
   # restore previous model if there is one
   if ckpt and ckpt.model_checkpoint_path:
      print "Restoring previous model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         pass
   '''
   if LOAD_MODEL:
      ckpt = tf.train.get_checkpoint_state(LOAD_MODEL)
      print "Restoring model..."
      try:
         saver.restore(sess, ckpt.model_checkpoint_path)
         print "Model restored"
      except:
         print "Could not restore model"
         raise
   '''


#### ######################################## training portion  ###########################
   step = sess.run(global_step)
   coord = tf.train.Coordinator()
   threads = tf.train.start_queue_runners(sess, coord=coord)
   merged_summary_op = tf.summary.merge_all()
   start = time.time()
   
   while True:
      epoch_num = step/(num_train/BATCH_SIZE)
      s = time.time()

      err, summary = sess.run([gen_loss, merged_summary_op])
      sess.run(train_op)
      step += 1
      summary_writer.add_summary(summary, step)
      
      if step % 4 == 0:      
	  print 'step:',step,'err:',err,'time:',time.time()-s

      if step % 100 == 0:
	  print
	  print 'Saving model'
	  print
	  saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
	  saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')



   print 'Finished training', time.time()-start
   saver.save(sess, EXPERIMENT_DIR+'checkpoint-'+str(step))
   saver.export_meta_graph(EXPERIMENT_DIR+'checkpoint-'+str(step)+'.meta')
   exit()
