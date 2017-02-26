'''

Cameron Fabbri

Script to convert all images in celeba with the following:

- Resize to 256x256
- Use skimage to create grayscale versions
- Convert to LAB colorspace
- Normalize to range [-1 1] for use in GANs

This does NOT replace images here. It puts them in my home data directory for use

'''

from scipy import misc
from skimage import color
from skimage.transform import resize
from tqdm import tqdm
import fnmatch
import sys
import os
import numpy as np
import cv2

if __name__ == '__main__':

   data_dir = sys.argv[1]

   pattern   = '*.jpg'
   image_list = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_list.append(os.path.join(d,filename))

   for image in tqdm(image_list):

      # get name
      image_name = os.path.basename(image).split('.')[0]

      # read image
      color_img = misc.imread(image)

      # convert to grayscale
      gray_img = color.rgb2gray(color_img)

      # resize both to 256x256
      color_img = resize(color_img, (256, 256))
      gray_img  = resize(gray_img, (256, 256))

      # convert color to LAB colorspace
      color_img = color.rgb2lab(color_img)

      # scale both to [-1, 1]
      color_img = color_img/127.5 -1.
      gray_img  = gray_img/127.5 -1.

      print color_img
      #misc.imsave('temp.jpg', color_img)
      cv2.imwrite('temp.jpg', np.float32(color_img))
      #new_img = misc.imread('temp.jpg')
      new_img = cv2.imread('temp.jpg')
      print new_img
      exit()
      # save to my data dir
      #np.save('/home/fabbric/data/images/celeba/color/'+image_name, color_img)
      #np.save('/home/fabbric/data/images/celeba/gray/'+image_name, gray_img)
