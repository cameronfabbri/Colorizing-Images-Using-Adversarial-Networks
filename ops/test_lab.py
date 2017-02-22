from skimage import io, color
import cv2

img = cv2.imread('img.jpg')
lab = color.rgb2lab(img)

image = color.lab2rgb(lab)


image = color.lab2rgb(image)
io.imsave('converted.png', image)

