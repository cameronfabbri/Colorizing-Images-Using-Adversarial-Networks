import cPickle as pickle
import sys
pkl_file = open(sys.argv[1], 'rb')
exp_info = pickle.load(pkl_file)

print
print 'LOSS_METHOD:     ',exp_info['LOSS_METHOD']
print 'DATA_DIR:        ',exp_info['DATA_DIR']
print 'GAN_EPOCHS:      ',exp_info['GAN_EPOCHS']
print 'PRETRAIN_EPOCHS: ',exp_info['PRETRAIN_EPOCHS']
print 'NUM_CRITIC:      ',exp_info['NUM_CRITIC']
print 'BATCH_SIZE:      ',exp_info['BATCH_SIZE']
print 'DATASET:         ',exp_info['DATASET']
print 'GAN_LR:          ',exp_info['GAN_LR']
print 'ARCHITECTURE:    ',exp_info['ARCHITECTURE']
print 'NUM_GPU:         ',exp_info['NUM_GPU']
print 'PRETRAIN_LR:     ',exp_info['PRETRAIN_LR']

