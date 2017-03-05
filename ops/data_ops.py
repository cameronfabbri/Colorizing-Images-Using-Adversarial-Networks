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
import config
import cPickle as pickle

trainData = collections.namedtuple('trainData', 'paths, inputs, targets, count, steps_per_epoch')

batch_size = config.batch_size

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

def getPaths(data_dir, ext='jpg'):
   pattern   = '*.'+ext
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(d,filename))
   return image_paths


def loadTrainData(input_dir, dataset):

   if input_dir is None or not os.path.exists(input_dir):
      raise Exception('input_dir does not exist')

   # get train/test sets
   if dataset == 'celeba':
      
      pkl_train_file = 'celeba_train.pkl'
      pkl_test_file  = 'celeba_test.pkl'

      if os.path.isfile(pkl_train_file) and os.path.isfile(pkl_test_file):
         print 'Found pickle file'
         train_paths = pickle.load(open(pkl_train_file, 'rb'))
         test_paths  = pickle.load(open(pkl_test_file, 'rb'))
      else:
         image_paths = getPaths(input_dir)
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

   input_paths = train_paths
   decode = tf.image.decode_jpeg

   if len(input_paths) == 0:
      raise Exception('input_dir contains no image files')

   with tf.name_scope('load_images'):
      path_queue = tf.train.string_input_producer(input_paths, shuffle='train')
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
   scale_size = 286
   CROP_SIZE = 256
   seed = random.randint(0, 2**31 - 1)
   def transform(image):
      r = image
      if flip:
         r = tf.image.random_flip_left_right(r, seed=seed)

      # area produces a nice downscaling, but does nearest neighbor for upscaling
      # assume we're going to be doing downscaling here
      r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)
      offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
      if scale_size > CROP_SIZE:
         r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
      elif scale_size < CROP_SIZE:
         raise Exception('scale size cannot be less than crop size')
      return r

   with tf.name_scope('input_images'):
      input_images = transform(inputs)

   with tf.name_scope('target_images'):
      target_images = transform(targets)

   paths_batch, inputs_batch, targets_batch = tf.train.batch([paths, input_images, target_images], batch_size=batch_size)
   steps_per_epoch = int(math.ceil(len(input_paths) / batch_size))

   return train_paths, test_paths, trainData(
      paths=paths_batch,
      inputs=inputs_batch,
      targets=targets_batch,
      count=len(input_paths),
      steps_per_epoch=steps_per_epoch,
   )


'''
def getBatch(batch_size, data, dataset, use_labels):

   label_size = 1000
   color_image_batch = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
   gray_image_batch  = np.empty((batch_size, 256, 256, 1), dtype=np.float32)

   if use_labels:
      label_batch = np.empty((batch_size, label_size), dtype=np.float32)

   random_imgs = random.sample(data, batch_size)
   for i, image_path in enumerate(random_imgs):
   
      if use_labels:
         label = np.zeros(label_size)
         label[int(image_path[1])] = 1
      
         image_path = image_path[0]

      # read in image
      color_img = misc.imread(image_path)

      # if it isn't 256x256 then resize it
      height, width, channels = color_img.shape
      if height or width is not 256:
         color_img = misc.imresize(color_img, (256, 256))
         height, width, channels = color_img.shape

      # convert rgb image to lab
      try: color_img = color.rgb2lab(color_img)
      except: continue # this happens if an original image is already gray

      # the gray image is just the first channel in the LAB image (lightness)
      gray_img = color_img[0]
      gray_img = misc.imresize(gray_img, (256, 256))
      gray_img = np.expand_dims(gray_img, 2)

      # scale to [-1, 1] range
      color_img = normalizeImage(color_img)
      gray_img  = normalizeImage(gray_img)

      color_image_batch[i, ...] = color_img
      gray_image_batch[i, ...]  = gray_img

      if use_labels:
         label_batch[i, ...] = label

   if use_labels: return color_image_batch, gray_image_batch, label_batch
   
   return color_image_batch, gray_image_batch
'''
