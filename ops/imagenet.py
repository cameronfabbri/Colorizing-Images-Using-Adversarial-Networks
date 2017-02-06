import os
import fnmatch
import random
import pandas as pd
import numpy as np
import cv2

'''
   General imagenet class that has the option to use or not use the labels, as well as use or not use
   the gray images (for colorization)

   ILSVRC/
      224x224_images/
         CLS-LOC/
            color/
               test/
                  ILSVRC2012_test_00000001.JPEG
                  ...
               train/
                  n01440764/
                     n01440764_10026.JPEG
                     ...
               val/
                  ILSVRC2012_val_00000001.JPEG
                  ...
            gray/
               test/
                  ILSVRC2012_test_00000001.JPEG
                  ...
               train/
                  n01440764/
                     n01440764_10026.JPEG
                     ...
               val/
                  ILSVRC2012_val_00000001.JPEG
                  ...
'''


class imagenet:

   def __init__(self, data_dir='/home/fabbric/data/images/imagenet/ILSVRC/224x224_images/CLS-LOC/', split='train', gray=True, labels=False):

      print 'Loading data...'

      self.__pattern = '*.JPEG'
      self.__train_num = 0
      self.__test_num  = 0
      self.__val_num   = 0
      self.__gray      = gray

      if split == 'train' or split == 'all':
         self.__images_train_dir = data_dir+'color/train/'
         self.__train_image_list = []
         for self.__d, self.__s, self.__fList in os.walk(self.__images_train_dir):
            for self.__filename in self.__fList:
               if fnmatch.fnmatch(self.__filename, self.__pattern):
                  color_img = os.path.join(self.__d, self.__filename)
                  if gray:
                     gray_img  = color_img.replace('color', 'gray')
                     self.__train_image_list.append([color_img, gray_img])
                  else: self.__train_image_list.append(color_img)
         self.__train_num = len(self.__train_image_list)

      if split == 'test' or split == 'all':
         self.__images_test_dir  = data_dir+'color/test/'
         self.__test_image_list = []
         for self.__d, self.__s, self.__fList in os.walk(self.__images_test_dir):
            for self.__filename in self.__fList:
               if fnmatch.fnmatch(self.__filename, self.__pattern):
                  color_img = os.path.join(self.__d, self.__filename)
                  if gray:
                     gray_img  = color_img.replace('color', 'gray')
                     self.__test_image_list.append([color_img, gray_img])
                  else: self.__test_image_list.append(color_img)
         self.__test_num  = len(self.__test_image_list)

      if split == 'val' or split == 'all':
         self.__images_val_dir   = data_dir+'color/val/'
         self.__val_image_list = []
         for self.__d, self.__s, self.__fList in os.walk(self.__images_val_dir):
            for self.__filename in self.__fList:
               if fnmatch.fnmatch(self.__filename, self.__pattern):
                  color_img = os.path.join(self.__d, self.__filename)
                  if gray:
                     gray_img  = color_img.replace('color', 'gray')
                     self.__val_image_list.append([color_img, gray_img])
                  else: self.__val_image_list.append(color_img)
         self.__val_num   = len(self.__val_image_list)

      print 'Done Loading data\n'


   '''
      
      Inputs: String
         'train'
         'test'
         'val'
      Returns: Dataframe of all gray and color images for the desired split

   '''
   def get_split_images(self, split):
      if split == 'train': return self.__train_image_list
      if split == 'test':  return self.__test_image_list
      if split == 'val':   return self.__val_image_list
      raise ValueError('Incorrent split type',split)

   def get_all_gray_images(self, split):
      if split == 'train': return self.__train_df[1]
      if split == 'test':  return self.__test_df[1]
      if split == 'val':   return self.__val_df[1]
      raise ValueError('Incorrent split type',split)
   
   def get_all_color_images(self, split):
      if split == 'train': return self.__train_df[0]
      if split == 'test':  return self.__test_df[0]
      if split == 'val':   return self.__val_df[0]
      raise ValueError('Incorrent split type',split)

   def get_split_num(self, split):
      if split == 'train': return self.__train_num
      if split == 'test': return self.__test_num
      if split == 'val': return self.__val_num
      raise ValueError('Incorrent split type',split)

   def get_batch(self, batch_size, split):
      if split == 'train': self.__split_list = self.__train_image_list
      if split == 'test': self.__split_list  = self.__test_image_list
      if split == 'val': self.__split_list   = self.__val_image_list

      self.__split_list = random.sample(self.__split_list, batch_size)

      color_images = []
      if self.__gray: 
         gray_images  = []
         for color_image, gray_image in self.__split_list:
            color_images.append(cv2.imread(color_image))
            gray_images.append(cv2.imread(gray_image))
         return np.asarray(color_images), np.asarray(gray_images)
      else:
         for color_image in self.__split_list:
            color_images.append(cv2.imread(color_image))
         return np.asarray(color_images)



