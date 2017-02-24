from scipy import misc
from skimage import color
import cv2
import numpy as np

img = misc.imread('Image.jpg')
img = color.rgb2lab(img)
print type(img[0][0][0])
exit()
#gray = img[:,:,0]
img = 255*color.lab2rgb(img)

misc.imsave('conv.jpg', img)
#misc.imsave('gray.JPEG', gray)

