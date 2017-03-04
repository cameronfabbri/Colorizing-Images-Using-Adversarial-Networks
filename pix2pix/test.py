import tensorflow as tf
from architecture import netD, netG_encoder, netG_decoder
import sys
sys.path.insert(0, '../ops/')

import data_ops
import config
import numpy as np

data_dir = config.data_dir
dataset = config.dataset

examples = data_ops.load_examples('/home/fabbric/data/images/celeba/original/')
img = examples.inputs

t = examples.targets
sess = tf.Session()
coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess, coord=coord)

print sess.run(examples.inputs)
exit()

import scipy.misc as misc
misc.imsave('img.jpg', img)

