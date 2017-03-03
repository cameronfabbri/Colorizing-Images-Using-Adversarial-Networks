'''

Operations used for data management

MASSIVE help from https://github.com/affinelayer/pix2pix-tensorflow/blob/master/pix2pix.py

'''
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

def preprocess(image):
   with tf.name_scope('preprocess'):
      # [0, 1] => [-1, 1]
      return image * 2 -1

def deprocess(image):
   with tf.name_scope('deprocess'):
      # [-1, 1] => [0, 1]
      return (image+1)/2

def preprocess_lab(lab):
   with tf.name_scope('preprocess_lab'):
      #L_chan = lab[:,:,:,0]
      #L_chan = tf.expand_dims(L_chan, 3)
      #a_chan = lab[:,:,:,1]
      #b_chan = lab[:,:,:,2]
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
   assertion = tf.assert_equal(tf.shape(image)[-1], 3, message="image must have 3 color channels")
   with tf.control_dependencies([assertion]):
      image = tf.identity(image)

   if image.get_shape().ndims not in (3, 4):
      raise ValueError("image must be either 3 or 4 dimensions")

   # make the last dimension 3 so that you can unstack the colors
   shape = list(image.get_shape())
   shape[-1] = 3
   image.set_shape(shape)
   return image

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
   image_list = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))
   return image_list


def _read_input(filename_queue):
   class DataRecord(object):
      pass
   reader             = tf.WholeFileReader()
   key, value         = reader.read(filename_queue)
   record             = DataRecord()
   decoded_image      = tf.image.decode_jpeg(value, channels=3)
   decoded_image_4d   = tf.expand_dims(decoded_image, 0)
   resized_image      = tf.image.resize_bilinear(decoded_image_4d, [256, 256])
   record.input_image = tf.squeeze(resized_image, squeeze_dims=[0])
   return record


def read_input_queue(filename_queue):
   read_input = _read_input(filename_queue)
   num_preprocess_threads = 8
   min_queue_examples = int(0.1 * 100)
   print("Shuffling")
   input_image = tf.train.shuffle_batch([read_input.input_image],
                                        batch_size=config.batch_size,
                                        num_threads=num_preprocess_threads,
                                        capacity=min_queue_examples + 8 * config.batch_size,
                                        min_after_dequeue=min_queue_examples)

   input_image = rgb_to_lab(input_image/127.5 - 1.)
   input_image = preprocess_lab(input_image)
   L_image, a_chan, b_chan = input_image

   a_chan = tf.expand_dims(a_chan, 3)
   b_chan = tf.expand_dims(b_chan, 3)

   ab_image = tf.concat([a_chan, b_chan], 3)
   
   return L_image, ab_image


def load_data2(data_dir, dataset):
   if dataset == 'celeba':
      train_paths = getPaths(data_dir+'train/')
      test_paths  = getPaths(data_dir+'test/')
   elif dataset == 'imagent':
      train_paths = getPaths(data_dir+'train/', ext='JPEG')
      test_paths  = getPaths(data_dir+'test/', ext='JPEG')

   return train_paths, test_paths


def load_data(data_dir, dataset):

   if dataset == 'celeba':
      train_paths = getPaths(data_dir+'train/')
      test_paths  = getPaths(data_dir+'test/')
   elif dataset == 'imagent':
      train_paths = getPaths(data_dir+'train/', ext='JPEG')
      test_paths  = getPaths(data_dir+'test/', ext='JPEG')

   decode = tf.image.decode_jpeg
   
   with tf.name_scope('load_images'):
      path_queue = tf.train.string_input_producer(train_paths)

      reader = tf.WholeFileReader()
      paths, contents = reader.read(path_queue)
      raw_input_ = decode(contents)
      raw_input_ = tf.image.convert_image_dtype(raw_input_, dtype=tf.float32)

      assertion = tf.assert_equal(tf.shape(raw_input_)[2], 3, message='image does not have 3 channels')

      with tf.control_dependencies([assertion]): raw_input_ = tf.identity(raw_input_)

      raw_input_.set_shape([None, None, 3])

      lab = rgb_to_lab(raw_input_)
      L_chan, a_chan, b_chan = preprocess_lab(lab)
      a_images = tf.expand_dims(L_chan, axis=2)
      b_images = tf.stack([a_chan, b_chan], axis=2)
      
      inputs, targets = [a_images, b_images]

      seed = random.randint(0, 2**31 - 1)

      def transform(image):
         r = image
         r = tf.image.random_flip_left_right(r, seed=seed)
         r = tf.image.resize_images(r, [256, 256], method=tf.image.ResizeMethod.AREA)
         return r

      with tf.name_scope('input_images'):
         input_images = transform(inputs)

      with tf.name_scope('target_images'):
         target_images = transform(targets)

      paths_batch, inputs_batch, targets_batch = tf.train.batch([
                                                paths,
                                                input_images,
                                                target_images],
                                                batch_size=config.batch_size)

      Data = collections.namedtuple('Data', 'paths, inputs, targets, count')
      return Data(
         paths=paths_batch,
         inputs=inputs_batch,
         targets=targets_batch,
         count=len(train_paths)
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
