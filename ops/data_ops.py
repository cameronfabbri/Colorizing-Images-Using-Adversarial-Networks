'''

Operations used for data management

'''
from scipy import misc
from skimage import color
import tensorflow as tf
import numpy as np
import math
import time
import random


def getBatch(batch_size, data, dataset, labels):

   if dataset == 'imagenet': label_size = 1000

   color_image_batch = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
   gray_image_batch  = np.empty((batch_size, 256, 256, 1), dtype=np.float32)
   label_batch       = np.empty((batch_size, label_size), dtype=np.float32)

   s = time.time()
   for i in range(batch_size):

      image_path = data[i][0]
      label = np.zeros(label_size)
     
      label[int(data[i][1])] = 1

      # read in image
      color_img = misc.imread(image_path)
 
      # convert rgb image to lab
      try:
         color_img = color.rgb2lab(color_img)
      except:
         print image_path
         exit()

      gray_img  = color.rgb2gray(color_img)
      color_img = misc.imresize(color_img, (256, 256))
      gray_img  = misc.imresize(gray_img, (256, 256))
      gray_img  = np.expand_dims(gray_img, 2)

      # scale to [-1, 1] range
      color_img = color_img/127.5 - 1.
      gray_img  = gray_img/127.5 - 1.

      color_image_batch[i, ...] = color_img
      gray_image_batch[i, ...]  = gray_img

      label_batch[i, ...] = label

   print time.time()-s
   return color_image_batch, gray_image_batch, label_batch

def unnormalizeImage(img, n='tanh'):
   if n == 'tanh':
      #img = (img+1.)/2.
      #img *= np.uint8(255.0/img.max())
      return (img+1)*127.5
   if n == 'norm': return np.uint8(255.0*img)

def normalizeImage(img, n='tanh'):
   if n == 'tanh': return img/127.5 - 1. # normalize between -1 and 1
   if n == 'norm': return img/255.0      # normalize between 0 and 1

'''
   unnormalize
   convert to uint8
   save
'''
def saveImage(img, step, dataset, n='tanh'):
   i = 0
   for image in img:
      image = unnormalizeImage(image)
      image = np.float64(image)
      image = color.lab2rgb(np.uint8(image))
      misc.imsave('images/'+dataset+'/'+step+'_'+str(i)+'.jpg', image)
      i += 1

'''
# read image
image = misc.imread('Image.jpg')

# convert to lab
image = color.rgb2lab(image)
# scale to [-1, 1]
image = normalizeImage(np.float32(image))

# scale back away from [-1, 1]
image = unnormalizeImage(image)

# lab2rgb takes in float64, will NOT work with float32
image = np.float64(image)
image = color.lab2rgb(image)

misc.imsave('conv.jpg', image)
'''
