import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import fnmatch
import cPickle as pickle

def load(
         data_dir='/home/fabbric/data/images/celeba/64x64_images/',
         normalize_fn='tanh',
         load=False,
         normalize=True,
         crop=False
   ):

   pkl_file = data_dir+'celeba.pkl'

   # first, check if a pickle file has been made with the image paths
   if os.isfile(pkl_file):
      print 'Pickle file found'
      # load images from pickle file, or just use the image list if load=false
      image_list == pickle.load(open(pkl_file, 'rb'))
      if load is False: return image_list
   else:
      # gather all image paths and set numpy array size
      pattern = "*.jpg"
      image_list = list()
      for d, s, fList in os.walk(data_dir):
         for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
               image_list.append(os.path.join(d,filename))
      pickle.dump(open(pkl_file, 'wb'), image_list)


   num_images = len(image_list)

   # load into memory. At (224,224) the most we can load is ~150,000 images (224*224*3*3*130000) bytes to gb
   if load is True: image_data = np.empty((num_images, 64, 64, 3), dtype=np.float32)

   print 'Loading data...'
   i = 0
   for image in tqdm(image_list):
      img = cv2.imread(image).astype('float32')
      img = img/127.5 - 1. # normalize between -1 and 1
      image_data[i, ...] = img
      i += 1
      #if i == 50000: break

   return image_data
