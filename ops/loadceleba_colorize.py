import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import fnmatch
import cPickle as pickle
from skimage import color
from data_ops import normalizeImage
import random

'''
   Important! This script assumes you point the data_dir to the color directory,
   and that there are images with the same name in ../gray/ like:

   /path/color/image1.jpg
   /path/gray/image1.jpg

   This is so we don't have to count all files twice
'''



'''
   Inputs: An image, either 2D or 3D

   Returns: The image cropped to 'size' from the center
'''
def centerCrop(img, size=64):
   height, width, c = img.shape
   center = (height/2, width/2)
   size   = size/2
   return img[center[0]-size:center[0]+size,center[1]-size:center[1]+size, :]
   
'''
   Inputs: A directory containing images (can have nested dirs inside) and optional extension

   Outputs: A list of image paths
'''
def getPaths(data_dir, ext='jpg'):
   pattern   = '*.'+ext
   image_list = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))
   return image_list


'''
   Loads the celeba data
'''
def load(
         data_dir='/home/fabbric/data/images/celeba/256x256_images/color/',
         normalize_fn='tanh',
         load=False,
         crop=True,
   ):
   
   # pickle file contains: data['color_images'] = ['/path/color/img1.jpg', '/path/color/img2.jpg', ... ]
   # pickle file contains: data['gray_images'] = ['/path/gray/img1.jpg', '/path/gray/img2.jpg', ... ]
   pkl_file = data_dir+'color_celeba.pkl'

   # first, check if a pickle file has been made with the image paths
   if os.path.isfile(pkl_file):
      print 'Pickle file found'
      image_paths = pickle.load(open(pkl_file, 'rb'))
      if load is False: return image_paths
   else:
      print 'Getting paths!'
      
      image_paths = dict()
      train_gray_images = []
      test_gray_images = []
      
      all_color_images = getPaths(data_dir)
      random.shuffle(all_color_images)
      
      image_paths['color_images_train'] = all_color_images[:190000]
      image_paths['color_images_test']  = all_color_images[190000:]

      for cimg in image_paths['color_images_train']:
         train_gray_images.append(cimg.replace('color', 'gray'))
      
      for cimg in image_paths['color_images_test']:
         test_gray_images.append(cimg.replace('color', 'gray'))

      image_paths['gray_images_train']  = train_gray_images
      image_paths['gray_images_test']   = test_gray_images
      
      pf   = open(pkl_file, 'wb')
      data = pickle.dumps(image_paths)
      pf.write(data)
      pf.close()
      if not load: return image_paths

   '''
   num_images = len(image_paths['color_images'])
   print num_images,'images'
   
   color_image_data = np.empty((num_images, 64, 64, 3), dtype=np.float32)
   gray_image_data  = np.empty((num_images, 64, 64, 1), dtype=np.float32)

   print 'Loading data...'
   i = 0
   for color_image, gray_image in tqdm(zip(image_paths['color_images'], image_paths['gray_images'])):

      color_img = centerCrop(cv2.imread(color_image).astype('float32'))
      gray_img  = centerCrop(cv2.imread(gray_image).astype('float32'))[:,:,0] # gray image is 3 dims of the same thing

      gray_img = np.expand_dims(gray_img, 2)

      color_img = normalizeImage(color_img, n=normalize_fn)
      gray_img  = normalizeImage(gray_img, n=normalize_fn)

      color_image_data[i, ...] = color_img
      gray_image_data[i, ...]  = gray_img
      
      i += 1
      if i == 100: break
   
   return color_image_data, gray_image_data
   '''