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

   if labels and dataset == 'imagenet': label_size = 1000
   if labels and dataset == 'lsun':     label_size = 7

   color_image_batch = np.empty((batch_size, 256, 256, 3), dtype=np.float32)
   gray_image_batch  = np.empty((batch_size, 256, 256, 1), dtype=np.float32)

   if labels:
      label_batch = np.empty((batch_size, label_size), dtype=np.float32)

   random_imgs = random.sample(data, batch_size)
   for i, image_path in enumerate(random_imgs):
   
      if labels:
         label = np.zeros(label_size)
         label[int(image_path[1])] = 1
      
         image_path = image_path[0]

      # read in image
      color_img = misc.imread(image_path)
      color_img = misc.imresize(color_img, (256, 256))

      # TODO Do NOT convert to GRAYSCALE --> the first channel in LAB color is gray!!

      # convert rgb image to lab
      try: color_img = color.rgb2lab(color_img)
      except: continue # this happens if an original image is already gray

      #gray_img  = color.rgb2gray(color_img)
      #print gray_img == color_img[0]

      gray_img = color_img[0]
      gray_img = misc.imresize(gray_img, (256, 256))
      gray_img = np.expand_dims(gray_img, 2)

      # scale to [-1, 1] range
      color_img = color_img/127.5 - 1.
      gray_img  = gray_img/127.5 - 1.

      color_image_batch[i, ...] = color_img
      gray_image_batch[i, ...]  = gray_img

      if labels:
         label_batch[i, ...] = label

   if labels: return color_image_batch, gray_image_batch, label_batch
   
   return color_image_batch, gray_image_batch

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
