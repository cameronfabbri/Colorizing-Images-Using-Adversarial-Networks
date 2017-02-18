import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import fnmatch

def load(data_dir='/home/fabbric/data/images/celeba/64x64_images/'):


   # gather all image paths and set numpy array size
   pattern = "*.jpg"
   image_list = list()
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))

   num_images = len(image_list)
   num_images = 100
   image_data = np.empty((num_images, 64, 64, 3), dtype=np.float32)

   print 'Loading data...'
   i = 0
   for image in tqdm(image_list):
      img = cv2.imread(image).astype('float32')
      img = 2*((img-np.min(img))/(np.max(img)-np.min(img))) - 1
      image_data[i, ...] = img
      i += 1
      if i == 100:
         break

   return image_data
