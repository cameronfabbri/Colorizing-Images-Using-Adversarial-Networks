import cPickle as pickle
import random
import numpy as np
import time
import cv2

f = open('lsun.pkl', 'rb')
data = pickle.load(f)

image_paths = []
image_paths += data['bedroom_train']



while True:
   print 'Getting batch'
   images = []
   s = time.time()
   batch_paths = random.sample(image_paths, 500)
   for path in batch_paths:
      images.append(cv2.imread(path))
   images = np.asarray(images)
   print time.time()-s

