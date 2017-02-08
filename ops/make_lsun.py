'''

Testing out speeds of making a one time pickle file containing the paths
of all of the images, then being able to grab from that.

'''
import os
import fnmatch
import cPickle as pickle

if __name__ == '__main__':

   root_dir = '/home/fabbric/data/images/lsun/data/'

   sub_folders = dict()
   sub_folders['train/conference_room_train/']  = 'conference_room_train'
   sub_folders['train/church_outdoor_train/']   = 'church_outdoor_train'
   sub_folders['train/dining_room_train/']      = 'dining_room_train'
   sub_folders['train/living_room_train/']      = 'living_room_train'
   sub_folders['train/restaurant_train/']       = 'restaurant_train'
   sub_folders['train/classroom_train/']        = 'classroom_train'
   sub_folders['train/kitchen_train/']          = 'kitchen_train'
   sub_folders['train/bedroom_train/']          = 'bedroom_train'
   sub_folders['train/bridge_train/']           = 'bridge_train'
   sub_folders['train/tower_train/']            = 'tower_train'
   sub_folders['test/'


   pattern = '*.webp'

   image_dict = dict()

   for folder in sub_folders:
      temp_list = []
      print 'Searching in', root_dir+folder
      for d, s, flist in os.walk(root_dir+folder):
         for filename in flist:
            if fnmatch.fnmatch(filename, pattern):
               image_path = os.path.join(d,filename)
               temp_list.append(image_path)
      image_dict[sub_folders[folder]] = temp_list
   pf = open('lsun.pkl', 'wb')
   data = pickle.dumps(image_dict)
   pf.write(data)
   pf.close()
   exit() 

