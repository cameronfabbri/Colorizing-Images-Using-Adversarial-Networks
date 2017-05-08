'''

Operations used for data management

MASSIVE help from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

'''

from __future__ import division
from __future__ import absolute_import

from scipy import misc
from skimage import color
import collections
import tensorflow as tf
import numpy as np
import math
import time
import random
import glob
import os
import fnmatch
import cPickle as pickle

Data = collections.namedtuple('trainData', 'paths, inputs, targets, count, steps_per_epoch')

def preprocess(image):
    with tf.name_scope('preprocess'):
        # [0, 1] => [-1, 1]
        return image * 2 - 1


def deprocess(image):
    with tf.name_scope('deprocess'):
        # [-1, 1] => [0, 1]
        return (image + 1) / 2


def preprocess_lab(lab):
    with tf.name_scope('preprocess_lab'):
        L_chan, a_chan, b_chan = tf.unstack(lab, axis=2)
        # L_chan: black and white with input range [0, 100]
        # a_chan/b_chan: color channels with input range ~[-110, 110], not exact
        # [0, 100] => [-1, 1],  ~[-110, 110] => [-1, 1]
        return [L_chan / 50 - 1, a_chan / 110, b_chan / 110]


def deprocess_lab(L_chan, a_chan, b_chan):
    with tf.name_scope('deprocess_lab'):
        # this is axis=3 instead of axis=2 because we process individual images but deprocess batches
        return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=3)
        #return tf.stack([(L_chan + 1) / 2 * 100, a_chan * 110, b_chan * 110], axis=2)


def augment(image, brightness):
    # (a, b) color channels, combine with L channel and convert to rgb
    a_chan, b_chan = tf.unstack(image, axis=3)
    L_chan = tf.squeeze(brightness, axis=3)
    lab = deprocess_lab(L_chan, a_chan, b_chan)
    rgb = lab_to_rgb(lab)
    return rgb



def check_image(image):
    assertion = tf.assert_equal(tf.shape(image)[-1], 3, message='image must have 3 color channels')
    with tf.control_dependencies([assertion]):
        image = tf.identity(image)

    if image.get_shape().ndims not in (3, 4):
        raise ValueError('image must be either 3 or 4 dimensions')

    # make the last dimension 3 so that you can unstack the colors
    shape = list(image.get_shape())
    shape[-1] = 3
    image.set_shape(shape)
    return image

# based on https://github.com/torch/image/blob/9f65c30167b2048ecbe8b7befdc6b2d6d12baee9/generic/image.c
def rgb_to_lab(srgb):
    with tf.name_scope('rgb_to_lab'):
        srgb = check_image(srgb)
        srgb_pixels = tf.reshape(srgb, [-1, 3])
        with tf.name_scope('srgb_to_xyz'):
            linear_mask = tf.cast(srgb_pixels <= 0.04045, dtype=tf.float32)
            exponential_mask = tf.cast(srgb_pixels > 0.04045, dtype=tf.float32)
            rgb_pixels = (srgb_pixels / 12.92 * linear_mask) + (((srgb_pixels + 0.055) / 1.055) ** 2.4) * exponential_mask
            rgb_to_xyz = tf.constant([
                #    X        Y          Z
                [0.412453, 0.212671, 0.019334], # R
                [0.357580, 0.715160, 0.119193], # G
                [0.180423, 0.072169, 0.950227], # B
            ])
            xyz_pixels = tf.matmul(rgb_pixels, rgb_to_xyz)

        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('xyz_to_cielab'):
            # convert to fx = f(X/Xn), fy = f(Y/Yn), fz = f(Z/Zn)

            # normalize for D65 white point
            xyz_normalized_pixels = tf.multiply(xyz_pixels, [1/0.950456, 1.0, 1/1.088754])

            epsilon = 6/29
            linear_mask = tf.cast(xyz_normalized_pixels <= (epsilon**3), dtype=tf.float32)
            exponential_mask = tf.cast(xyz_normalized_pixels > (epsilon**3), dtype=tf.float32)
            fxfyfz_pixels = (xyz_normalized_pixels / (3 * epsilon**2) + 4/29) * linear_mask + (xyz_normalized_pixels ** (1/3)) * exponential_mask

            # convert to lab
            fxfyfz_to_lab = tf.constant([
                #  l       a       b
                [  0.0,  500.0,    0.0], # fx
                [116.0, -500.0,  200.0], # fy
                [  0.0,    0.0, -200.0], # fz
            ])
            lab_pixels = tf.matmul(fxfyfz_pixels, fxfyfz_to_lab) + tf.constant([-16.0, 0.0, 0.0])

        return tf.reshape(lab_pixels, tf.shape(srgb))


def lab_to_rgb(lab):
    with tf.name_scope('lab_to_rgb'):
        lab = check_image(lab)
        lab_pixels = tf.reshape(lab, [-1, 3])
        # https://en.wikipedia.org/wiki/Lab_color_space#CIELAB-CIEXYZ_conversions
        with tf.name_scope('cielab_to_xyz'):
            # convert to fxfyfz
            lab_to_fxfyfz = tf.constant([
                #   fx      fy        fz
                [1/116.0, 1/116.0,  1/116.0], # l
                [1/500.0,     0.0,      0.0], # a
                [    0.0,     0.0, -1/200.0], # b
            ])
            fxfyfz_pixels = tf.matmul(lab_pixels + tf.constant([16.0, 0.0, 0.0]), lab_to_fxfyfz)

            # convert to xyz
            epsilon = 6/29
            linear_mask = tf.cast(fxfyfz_pixels <= epsilon, dtype=tf.float32)
            exponential_mask = tf.cast(fxfyfz_pixels > epsilon, dtype=tf.float32)
            xyz_pixels = (3 * epsilon**2 * (fxfyfz_pixels - 4/29)) * linear_mask + (fxfyfz_pixels ** 3) * exponential_mask

            # denormalize for D65 white point
            xyz_pixels = tf.multiply(xyz_pixels, [0.950456, 1.0, 1.088754])

        with tf.name_scope('xyz_to_srgb'):
            xyz_to_rgb = tf.constant([
                #     r           g          b
                [ 3.2404542, -0.9692660,  0.0556434], # x
                [-1.5371385,  1.8760108, -0.2040259], # y
                [-0.4985314,  0.0415560,  1.0572252], # z
            ])
            rgb_pixels = tf.matmul(xyz_pixels, xyz_to_rgb)
            # avoid a slightly negative number messing up the conversion
            rgb_pixels = tf.clip_by_value(rgb_pixels, 0.0, 1.0)
            linear_mask = tf.cast(rgb_pixels <= 0.0031308, dtype=tf.float32)
            exponential_mask = tf.cast(rgb_pixels > 0.0031308, dtype=tf.float32)
            srgb_pixels = (rgb_pixels * 12.92 * linear_mask) + ((rgb_pixels ** (1/2.4) * 1.055) - 0.055) * exponential_mask

        return tf.reshape(srgb_pixels, tf.shape(lab))

def getPaths(data_dir, gray_images=None, ext='jpg'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            fname_ = os.path.join(d,filename)
            if gray_images is not None:
               if fname_ in gray_images:
                  continue
            image_paths.append(fname_)
   return image_paths


# TODO add in files to exclude (gray ones)
def loadData(data_dir, dataset, batch_size, jitter=True, train=True, SIZE=256):

   if data_dir is None or not os.path.exists(data_dir):
      raise Exception('data_dir does not exist')

   if dataset == 'true_gray':
      train_paths = []
      test_paths  = pickle.load(open('/home/fabbric/Research/colorgans/files/true_gray/true_gray.pkl', 'rb'))

   if dataset == 'looneytunes':
      print 'Using Looney Tunes'
      pkl_train_file = 'files/looneytunes_train.pkl'
      pkl_test_file  = 'files/looneytunes_test.pkl'

      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
      else:
         train_dir = data_dir
         train_paths = getPaths(train_dir, gray_images=None)
         random.shuffle(train_paths)
         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()

   if dataset == 'stl10':
      print 'Using stl10'
      pkl_train_file = 'files/stl10_train.pkl'
      pkl_test_file  = 'files/stl10_test.pkl'

      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
         test_paths = test_paths[54:58]
      else:
         train_dir = data_dir+'train/'
         test_dir  = data_dir+'test/'
         train_paths = getPaths(train_dir, gray_images=None)
         test_paths  = getPaths(test_dir, gray_images=None)
         random.shuffle(train_paths)
         random.shuffle(test_paths)
         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
         pf   = open(pkl_test_file, 'wb')
         data = pickle.dumps(test_paths)
         pf.write(data)
         pf.close()
   if dataset == 'celeba':
      print 'Using celeba'
      pkl_train_file = 'files/celeba_train.pkl'
      pkl_test_file  = 'files/celeba_test.pkl'

      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
         # ADDED FOR TESTING ONLY
         #test_paths = test_paths[54:58]
         test_paths = test_paths[2354:2358]
      else:
         image_paths = getPaths(data_dir)
         random.shuffle(image_paths)

         train_paths = image_paths[:195000]
         test_paths  = image_paths[195000:]

         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
         
         pf   = open(pkl_test_file, 'wb')
         data = pickle.dumps(test_paths)
         pf.write(data)
         pf.close()
   if dataset == 'imagenet':
      print 'Using imagenet'
      pkl_train_file = 'files/imagenet_train.pkl'
      pkl_test_file  = 'files/imagenet_test.pkl'
      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file, loading data...'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
      else:
         #train_dir = data_dir+'train/'
         train_dir = data_dir
         test_dir  = data_dir
         #test_dir  = data_dir+'test/'
         #train_paths = getPaths(train_dir, ext='JPEG')
         train_paths = getPaths(train_dir,ext='png')
         test_paths  = getPaths(test_dir,ext='png')
         #test_paths  = getPaths(test_dir, ext='JPEG')
         random.shuffle(train_paths)
         random.shuffle(test_paths)
         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
         pf   = open(pkl_test_file, 'wb')
         data = pickle.dumps(test_paths)
         pf.write(data)
         pf.close()
   if dataset == 'places2_standard':
      gray_images = []
      with open('files/gray_images_standard.txt','r') as f:
         for line in f:
            line = line.rstrip()
            gray_images.append(line)
      print 'Using places2 standard'
      pkl_train_file = 'files/places2_standard_train.pkl'
      pkl_test_file  = 'files/places2_standard_test.pkl'

      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file, loading data...'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
      else:
         train_dir = data_dir+'train_256/'
         test_dir  = data_dir+'test_256/'
         train_paths = getPaths(train_dir, gray_images=gray_images)
         test_paths  = getPaths(test_dir, gray_images=gray_images)
         random.shuffle(train_paths)
         random.shuffle(test_paths)
         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
         pf   = open(pkl_test_file, 'wb')
         data = pickle.dumps(test_paths)
         pf.write(data)
         pf.close()
   if dataset == 'places2_challenge':
      print 'Using places2 challenge'
      pkl_train_file = 'files/places2_challenge_train.pkl'
      pkl_test_file  = 'files/places2_challenge_test.pkl'

      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file, loading data...'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
      else:
         train_dir = data_dir+'train_256/'
         test_dir  = data_dir+'test_256/'
         train_paths = getPaths(train_dir)
         test_paths  = getPaths(test_dir)
         random.shuffle(train_paths)
         random.shuffle(test_paths)
         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
         pf   = open(pkl_test_file, 'wb')
         data = pickle.dumps(test_paths)
         pf.write(data)
         pf.close()

   if dataset == 'bedroom':
      pkl_train_file = 'files/bedroom_train.pkl'
      pkl_test_file  = 'files/bedroom_test.pkl'
      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
      else:   
         print 'Using lsun subset bedroom'
         train_dir = data_dir+'images/bedroom/train/'
         test_dir  = data_dir+'images/bedroom/val/'
         train_paths = getPaths(train_dir)
         test_paths  = getPaths(test_dir)
         random.shuffle(train_paths)
         random.shuffle(test_paths)
         pf   = open(pkl_train_file, 'wb')
         data = pickle.dumps(train_paths)
         pf.write(data)
         pf.close()
         pf   = open(pkl_test_file, 'wb')
         data = pickle.dumps(test_paths)
         pf.write(data)
         pf.close()

   print 'Done!'
   if train: input_paths = train_paths
   else:
      #random.shuffle(test_paths)
      #input_paths = test_paths[123:127]
      input_paths = test_paths
   print len(input_paths),'images!'
   decode = tf.image.decode_image
   
   if len(input_paths) == 0:
      raise Exception('data_dir contains no image files')

   with tf.name_scope('load_images'):
      path_queue = tf.train.string_input_producer(input_paths, shuffle=True)
      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input = decode(contents)
      raw_input = tf.image.convert_image_dtype(raw_input, dtype=tf.float32)

      assertion = tf.assert_equal(tf.shape(raw_input)[2], 3, message='image does not have 3 channels')
      with tf.control_dependencies([assertion]):
         raw_input = tf.identity(raw_input)

      raw_input.set_shape([None, None, 3])

      # load color and brightness from image, no B image exists here
      lab = rgb_to_lab(raw_input)
      L_chan, a_chan, b_chan = preprocess_lab(lab)
      a_images = tf.expand_dims(L_chan, axis=2)
      b_images = tf.stack([a_chan, b_chan], axis=2)
    
   inputs, targets = [a_images, b_images]

   # synchronize seed for image operations so that we do the same operations to both
   # input and output images
   flip = 1
   scale_size = SIZE+20
   CROP_SIZE  = SIZE

   seed = random.randint(0, 2**31 - 1) 
   def transform(image):
      r = image
      r = tf.image.random_flip_left_right(r, seed=seed)

      # area produces a nice downscaling, but does nearest neighbor for upscaling
      # assume we're going to be doing downscaling here
      r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
      offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
      r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
      return r

   if train and jitter:
      with tf.name_scope('input_images'):
         input_images = transform(inputs)
      with tf.name_scope('target_images'):
         target_images = transform(targets)
   else:
      input_images = tf.image.resize_images(inputs, [CROP_SIZE, CROP_SIZE], method=tf.image.ResizeMethod.AREA)
      target_images = tf.image.resize_images(targets, [CROP_SIZE, CROP_SIZE], method=tf.image.ResizeMethod.AREA)

   paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
   steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

   return Data(
      paths=paths_batch,
      inputs=inputs_batch,
      targets=targets_batch,
      count=len(input_paths),
      steps_per_epoch=steps_per_epoch,
   )
