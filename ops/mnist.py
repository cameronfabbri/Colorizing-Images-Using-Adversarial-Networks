import cv2
import cPickle as pickle
import numpy as np


def load_mnist(mnist_file='/home/fabbric/data/images/mnist/mnist.pkl'):

   pf = open(mnist_file, 'rb')
   mnist = pickle.load(pf)

   train, test, val = mnist

   train_images = []
   test_images  = []
   val_images   = []

   train_images_ = train[0]
   val_images_   = test[0]
   test_images_  = val[0]

   train_labels = train[1]
   val_labels   = test[1]
   test_labels  = val[1]

   for train_img in train_images_:
      train_img = np.expand_dims(np.resize(train_img, (28,28)), 2)
      train_images.append(train_img.astype('float32'))
   
   for test_img in test_images_ :
      test_img  = np.expand_dims(np.resize(test_img, (28,28)), 2)
      test_images.append(test_img.astype('float32'))

   for val_img in val_images_:
      val_img   = np.expand_dims(np.resize(val_img, (28,28)), 2)
      val_images.append(val_img.astype('float32'))


   return np.asarray(train_images), np.asarray(test_images), np.asarray(val_images)

if __name__ == '__main__':

   train_imgs, test_imgs, val_imgs = load_mnist()

