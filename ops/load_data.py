'''

Cameron Fabbri

Script to load image paths and labels if they exist

'''

import xml.etree.ElementTree
import fnmatch
import ntpath
import os
import xmltodict
import cPickle as pickle
from tqdm import tqdm
import numpy as np

'''
   Imagenet data. Using the test split as testing when using labels because the test set
   does not have labels.
'''
#def loadImagenet(data_dir='/mnt/data2/images/imagenet/ILSVRC2016/CLS_LOC_dataset/Data/CLS-LOC/',
def loadImagenet(data_dir='/home/fabbric/data/images/imagenet/ILSVRC/original/Data/CLS-LOC/',
                  use_labels=True,
                  split = 'train'):

   train_dir = data_dir + 'train/'
   test_dir  = data_dir + 'val/'

   # labels for testidation
   #test_anno = '/mnt/data2/images/imagenet/ILSVRC2016/CLS_LOC_dataset/Annotations/CLS-LOC/val/'
   test_anno = '/home/fabbric/data/images/imagenet/ILSVRC/original/Annotations/CLS-LOC/val/'

   # dictionary containing label to one hot vector location

   if split == 'test': imagenet_pkl = train_dir+'imagenet_val_'+str(use_labels)+'.pkl'
   if split == 'train': imagenet_pkl = train_dir+'imagenet_train_'+str(use_labels)+'.pkl'

   if split == 'train':
      image_dir = train_dir
   elif split == 'test':
      image_dir = test_dir

   if os.path.isfile(imagenet_pkl):
      print 'Found pickle file, loading...'
      image_list = pickle.load(open(imagenet_pkl, 'rb'))
      return image_list
   else:
      label_dict = dict()
      with open(data_dir+'map_clsloc.txt', 'r') as f:
         for line in f:
            line = line.rstrip().split()
            label_dict[line[0]] = int(line[1])-1

      pattern = '*.JPEG'
      image_list = []
      label_list = []
      label = None
      for d, s, fList in tqdm(os.walk(image_dir)):
         for filename in fList:
            if fnmatch.fnmatch(filename, pattern):
               image_path = os.path.join(d, filename)
               if use_labels and split == 'train':
                  label_str = ntpath.basename(image_path).split('_')[0]
               elif use_labels and split == 'test':
                  label_file = test_anno+ntpath.basename(image_path).split('.JPEG')[0]+'.xml'
                  with open(label_file) as fd:
                     doc = xmltodict.parse(fd.read())
                  try: label_str = doc['annotation']['object']['name']
                  except: label_str= doc['annotation']['object'][0]['name']
               
               label = label_dict[label_str]
               image_list.append([image_path, label])

      image_list = np.asarray(image_list)

      print 'Done....writing file'
      pf = open(imagenet_pkl, 'wb')
      data = pickle.dumps(image_list)
      pf.write(data)
      pf.close()
      print 'Done'

   return image_list

def load(dataset, use_labels, split):
   if dataset == 'imagenet': return loadImagenet(use_labels=use_labels, split=split)
   elif dataset == 'celeba': return loadCeleba(use_labels=use_labels)
   elif dataset == 'lsun'  : return loadLsun(use_labels=use_labels)
   elif dataset == 'sun'   : return loadSun(use_labels=use_labels)

if __name__ == '__main__':

   test_image_list   = load('imagenet', True, 'test')
   train_image_list = load('imagenet', True, 'train')

   print len(train_image_list)
   print len(test_image_list)
   

