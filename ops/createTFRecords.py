'''

Cameron Fabbri

'''

from skimage import color, img_as_float, io
from skimage.transform import resize
from tqdm import tqdm
import fnmatch
import sys
import os
import numpy as np
import cv2
import time
import random
import cPickle as pickle
import tensorflow as tf

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def _int64_feature(value):
   return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# helper function
def _bytes_feature(value):
   return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

if __name__ == '__main__':

   data_dir = sys.argv[1]

   pkl_file = data_dir+'data.pkl'

   if os.path.isfile(pkl_file):
      print 'Pickle file found!'
      data = pickle.load(open(pkl_file, 'rb'))
      train_list = data['train_list']
      test_list  = data['test_list']
   else:
      pattern   = '*.jpg'
      image_list = []
      for d, s, fList in os.walk(data_dir):
         for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
               image_list.append(os.path.join(d,filename))
      # split up train and test
      total_images = len(image_list)
      train_per    = .95
      test_per     = .05
      train_num    = int(train_per*total_images)
      random.shuffle(image_list)   
      train_list = image_list[:train_num]
      test_list  = image_list[train_num:]
      image_paths = dict()
      image_paths['train_list'] = train_list
      image_paths['test_list']  = test_list
      data = pickle.dumps(image_paths)
      pf = open(pkl_file, 'wb')
      pf.write(data)
      pf.close()

   train_writer = tf.python_io.TFRecordWriter('/home/fabbric/data/images/celeba/celeba_train.tfrecords')
   test_writer = tf.python_io.TFRecordWriter('/home/fabbric/data/images/celeba/celeba_test.tfrecords')

   # writing all train files
   i = 0
   for image in tqdm(train_list):
      # get name
      image_name = os.path.basename(image).split('.')[0]
      # read image
      color_img = io.imread(image)
      # convert to grayscale
      gray_img = color.rgb2gray(color_img)
      # resize both to 256x256
      color_img = resize(color_img, (256, 256))
      gray_img  = resize(gray_img, (256, 256))
      # convert color to LAB colorspace
      color_img = color.rgb2lab(color_img)
      # scale to [-1 1] tanh range
      color_img = color_img/127.5 -1.
      gray_img  = gray_img/127.5 -1.
      gray_img  = np.expand_dims(gray_img, 2)
      color_img = color_img.flatten()
      gray_img  = gray_img.flatten()
      color_img = np.float32(color_img)
      gray_img  = np.float32(gray_img)
      example   = tf.train.Example(features=tf.train.Features(feature={
         'color_image': _bytes_feature(color_img.tostring()),
         'gray_image':  _bytes_feature(gray_img.tostring())}))
      train_writer.write(example.SerializeToString())
      i += 1
   train_writer.close()

   # writing all test files
   i = 0
   for image in tqdm(test_list):
      # get name
      image_name = os.path.basename(image).split('.')[0]
      # read image
      color_img = io.imread(image)
      # convert to grayscale
      gray_img = color.rgb2gray(color_img)
      # resize both to 256x256
      color_img = resize(color_img, (256, 256))
      gray_img  = resize(gray_img, (256, 256))
      # convert color to LAB colorspace
      color_img = color.rgb2lab(color_img)
      # scale to [-1 1] tanh range
      color_img = color_img/127.5 -1.
      gray_img  = gray_img/127.5 -1.
      gray_img  = np.expand_dims(gray_img, 2)
      color_img = color_img.flatten()
      gray_img  = gray_img.flatten()
      color_img = np.float32(color_img)
      gray_img  = np.float32(gray_img)
      example   = tf.train.Example(features=tf.train.Features(feature={
         'color_image': _bytes_feature(color_img.tostring()),
         'gray_image':  _bytes_feature(gray_img.tostring())}))
      test_writer.write(example.SerializeToString())
      i += 1
   test_writer.close()

