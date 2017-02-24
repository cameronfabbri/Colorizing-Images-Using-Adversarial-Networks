'''

Operations used for data management

'''
from scipy import misc
from skimage import color
import tensorflow as tf
import numpy as np
import math

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
