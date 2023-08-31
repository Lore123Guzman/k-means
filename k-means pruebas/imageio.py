import numpy as np
import imageio
import scipy.ndimage
import cv2

img="FOTO.jpg"

def grayscale(rgb):
    return np.dot(rgb[...,:3],[0.299,0.587,0,114])

def dodge(front,back):
    result=front*250/(250-back)
    result[result>250]=250
    result[back==255]=255
    return result.astype('uint8')



h=imageio.imread(img)
i=grayscale(h)
r=255-i

b=scipy.ndimage.filters.gaussian_filter(i,sigma=10)
r=dodge(b,h)

cv2.imwrite('1.png',r)
    