import os
import fnmatch
import cPickle as pickle


data_dir = '/home/fabbric/Research/colorgans/files/true_gray/gray_crop/'

pattern   = '*.png'
image_paths = []
for d, s, fList in os.walk(data_dir):
   for filename in fList:
      if fnmatch.fnmatch(filename, pattern):
         fname_ = os.path.join(d,filename)
         image_paths.append(fname_)

pf   = open('true_gray.pkl', 'wb')
data = pickle.dumps(image_paths)
pf.write(data)
pf.close()

