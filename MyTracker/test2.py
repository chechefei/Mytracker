import cv2
from pyhog import pyhog
import numpy as np

image = cv2.imread('00000001.jpg')/255.
hog_feature_t = pyhog.features_pedro(image, 4)
print(hog_feature_t.shape)
img_crop = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
img_ = np.around(img_crop,decimals = 4)
np.savetxt('fhog1.txt',img_crop[:,:,1])
print(img_crop[:,:,0].shape)