import cv2
import pandas as pd
import cPickle as pickle
import pprint
import numpy as np

class mnist:


   def __init__(self, mnist_file='/home/fabbric/data/images/mnist/mnist.pkl'):

      self.__mnist = pd.read_pickle(mnist_file)

      # train, val, test
      self.__df = pd.DataFrame(list(self.__mnist), columns=['images', 'labels'])

      self.__images = self.__df['images']
      self.__labels = self.__df['labels']
      
      self.__train_images = self.__images[0]
      self.__val_images   = self.__images[1]
      self.__test_images  = self.__images[2]

      self.__train_labels = self.__labels[0]
      self.__val_labels   = self.__labels[1]
      self.__test_labels  = self.__labels[2]


   def get_train_images(self):
      return self.__train_images

   def get_val_images(self):
      return self.__val_images

   def get_test_images(self):
      return self.__test_images

   def get_train_labels(self):
      return self.__train_labels
   
   def get_val_labels(self):
      return self.__val_labels
   
   def get_test_labels(self):
      return self.__test_labels

obj = mnist()

print 'train images:', obj.get_train_images().shape
print 'val images:', obj.get_val_images().shape
print 'test images:', obj.get_test_images().shape
print
print 'train labels:', obj.get_train_labels().shape
print 'val labels:', obj.get_val_labels().shape
print 'test labels:', obj.get_test_labels().shape

