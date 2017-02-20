import cv2
import numpy as np
from random import shuffle
from tqdm import tqdm
import os
import fnmatch
import cPickle as pickle

'''
   resizes image to 96x96 then crops 64x64 center
'''
def crop_(img):
   #height, width, channels = img.shape
   #if height is not 96 and width is not 96: img = cv2.resize(img, (96,96))
   height, width, channels = img.shape
   center = (height/2, width/2)
   img = img[center[0]-32:center[0]+32,center[1]-32:center[1]+32, :]
   return img

def getPaths(data_dir):
   pattern   = "*.jpg"
   image_list = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))
   return image_list

def load(
         data_dir='/home/fabbric/data/images/celeba/96x96_images/',
         normalize_fn='tanh',
         normalize=True,
         load=False,
         crop=True,
         gray=False
   ):
   
   '''

      gray_celeba.pkl contains color AND gray paths to images.
      celeba.pkl just contains color image paths. Use gray
      if doing colorization

      gray pickle file contains: data['color_images'] = ['img1.jpg', 'img2.jpg', ... ]
                                 data['gray_images']  = ['img1.jpg', 'img2.jpg', ... ]

      celeba pickle file contains: data['images'] = ['img1.jpg', 'img2.jpg', ... ]

   '''
   if gray: pkl_file = data_dir+'gray_celeba.pkl'
   else: pkl_file    = data_dir+'celeba.pkl'

   # first, check if a pickle file has been made with the image paths
   if os.path.isfile(pkl_file):
      print 'Pickle file found'
      # load images from pickle file, or just use the image list if load=false
      image_paths = pickle.load(open(pkl_file, 'rb'))
      if load is False: return image_paths
   else:
      # if gray need to create two image lists, one color and one gray
      if gray:
         color_dir = data_dir+'color/'
         gray_dir  = data_dir+'gray/'
         
         color_images = getPaths(color_dir)
         gray_images  = getPaths(gray_dir)
         image_paths = dict()
         image_paths['color_images'] = color_images
         image_paths['gray_images']  = gray_images
         pickle.dump(open(pkl_file, 'wb'), image_paths)
      else:
         print 'getting paths!'
         image_dir = data_dir+'color/'
         images = getPaths(image_dir)
         image_paths = dict()
         image_paths['images'] = images
         pf   = open(pkl_file, 'wb')
         data = pickle.dumps(image_paths)
         pf.write(data)
         pf.close()
   
   if not load: return image_paths

   # TEMP TODO fix this, and also resize all images to 96x96 instead of the 64x64
   if not gray: num_images = len(image_paths['images'])
   # load into memory. At (224,224) the most we can load is ~150,000 images (224*224*3*3*130000) bytes to gb
   if load is True: image_data = np.empty((num_images, 64, 64, 3), dtype=np.float32)

   print 'Loading data...'
   i = 0
   for image in tqdm(image_paths['images']):
      img = cv2.imread(image).astype('float32')
      img = crop_(img)
      img = img/127.5 - 1. # normalize between -1 and 1
      image_data[i, ...] = img
      i += 1
      #if i == 1000: break

   return image_data
