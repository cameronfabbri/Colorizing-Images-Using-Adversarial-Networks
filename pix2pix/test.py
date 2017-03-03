import tensorflow as tf
from architecture import netD, netG_encoder, netG_decoder
import sys
sys.path.insert(0, '../ops/')

import data_ops
import config
import numpy as np

data_dir = config.data_dir
dataset = config.dataset

#Data      = data_ops.load_data(data_dir, dataset)
#num_train = Data.count
#L_image   = Data.inputs
#ab_image  = Data.targets

#L_image = data_ops.deprocess(L_image)
#targets = data_ops.augment(ab_image, L_image)
Data = data_ops.load_data(data_dir, dataset)
#Data = data_ops.load_data('/mnt/data2/images/imagenet/ILSVRC2016/CLS_LOC_dataset/Data/CLS-LOC/', 'imagenet')
img = Data.inputs
t = Data.targets
print t
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord=coord)

print sess.run(t)
exit()

import scipy.misc as misc
misc.imsave('img.jpg', img)

