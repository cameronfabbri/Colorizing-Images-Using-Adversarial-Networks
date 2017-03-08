# TODO 
# capitalize everything
# be able to override using --DATA_DIR etc
# create a checkpoint directory for each experiment with a pickle file
# fix the thing that makes me comment out with tf.device() when I don't wanna use a GPU
# shuffle testing
# figure out why testing is messed up
pretrain_epochs = 5 # if 0 then no pretrain
gan_epochs      = 5 # if 0 then no GAN at all
architecture    = 'colorarch'
loss_method     = 'wasserstein'
dataset         = 'celeba'
#data_dir        = '/home/fabbric/data/images/celeba/original/'
data_dir       = '/mnt/data2/images/celeba/images/'
learning_rate   = 2e-5
batch_size      = 32
multi_gpu       = True
