import cv2
import sys
import os
import fnmatch

def getPaths(data_dir):
   pattern   = '*.png'
   image_paths = []
   for d, s, fList in os.walk(data_dir):
      for filename in fList:
         if fnmatch.fnmatch(filename, pattern):
            image_paths.append(os.path.join(d,filename))
   return image_paths


data_dir = sys.argv[1]
images   = getPaths(data_dir)

for img in images:

   name = os.path.basename(img)
   folder = os.path.dirname(img)+'/'
   new_img = folder+name.split('.')[0]+'_gray.png'

   img = cv2.imread(img)
   img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
   cv2.imwrite(new_img, img)

