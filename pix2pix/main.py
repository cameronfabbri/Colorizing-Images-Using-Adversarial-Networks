import sys
import ntpath
from train import buildAndTrain
import os
import tensorflow as tf

def main(argv=None):

   # this loads a config file like: import config_name
   try:
      config_file = ntpath.basename(sys.argv[1]).split('.py')[0]
      config = __import__(config_file)
      print '\nsuccessfully imported',config_file
   except:
      print 'config',sys.argv[1],'not found'
      print
      raise
      exit()

   # set up params from config
   checkpoint_dir = config.checkpoint_dir
   learning_rate  = config.learning_rate
   batch_size     = config.batch_size
   dataset        = config.dataset
   use_labels     = config.use_labels

   if checkpoint_dir[-1] is not '/': checkpoint_dir+='/'

   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(checkpoint_dir+dataset+'_'+str(use_labels))
   except: pass
   try: os.mkdir('images/')
   except: pass
   try: os.mkdir('images/'+dataset+'_'+str(use_labels))
   except: pass
   
   checkpoint_dir = checkpoint_dir+dataset+'_'+str(use_labels)+'/'

   info = dict()
   info['checkpoint_dir'] = checkpoint_dir
   info['learning_rate']  = learning_rate
   info['batch_size']     = batch_size
   info['dataset']        = dataset
   info['use_labels']     = use_labels

   print
   print 'checkpoint_dir:',checkpoint_dir
   print 'learning_rate: ',learning_rate
   print 'batch_size:    ',batch_size
   print 'dataset:       ',dataset
   print 'use_labels:    ',use_labels
   print
   # build the graph - placeholders, loss functions, etc, then call train.
   buildAndTrain(info)

if __name__ == '__main__':
   tf.app.run()
