import os
import cv2
import numpy as np
import fnmatch

class lsun:


   def __init__(self, data_dir='/home/fabbric/data/images/lsun/data/', split='train', subset=None):


      data_dir += split + '/'
      if subset: data_dir += subset + '_train/'
     
      self.__patern = '*.webp'

      image_list = []
      
      for d, s, flist in os.walk(data_dir):
         for filename in flist:
            if fnmatch.fnmatch(filename, pattern):
               image_list.append(os.path.join(d,filename))



   def get_batch(self, batch_size, split='train', subset=None):

      return 1


obj = lsun(subset='bedroom')




