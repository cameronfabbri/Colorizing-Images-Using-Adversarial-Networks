import sys
import ntpath
from train import buildAndTrain
import os

if __name__ == '__main__':

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
   data_dir       = config.data_dir
   dataset        = config.dataset
   load           = config.load
   use_pt         = config.use_pt

   if checkpoint_dir[-1] is not '/': checkpoint_dir+='/'

   # TODO fix this. os.mkdirs is giving me trouble so just a temp fix
   try: os.mkdir(checkpoint_dir)
   except: pass
   try: os.mkdir(checkpoint_dir+dataset+'_'+str(use_pt))
   except: pass
   try: os.mkdir('images/')
   except: pass
   try: os.mkdir('images/'+dataset+'_'+str(use_pt))
   except: pass

   checkpoint_dir = checkpoint_dir+dataset+'_'+str(use_pt)+'/'

   info = dict()
   info['checkpoint_dir'] = checkpoint_dir
   info['learning_rate']  = learning_rate
   info['batch_size']     = batch_size
   info['data_dir']       = data_dir
   info['dataset']        = dataset
   info['use_pt']         = use_pt
   info['load']           = load

   print
   print 'checkpoint_dir:',checkpoint_dir
   print 'learning_rate: ',learning_rate
   print 'batch_size:    ',batch_size
   print 'data_dir:      ',data_dir
   print 'use_pt:        ',use_pt
   print 'dataset:       ',dataset
   print 'load:          ',load
   print
   
   # build the graph - placeholders, loss functions, etc, then call train.
   buildAndTrain(info)

