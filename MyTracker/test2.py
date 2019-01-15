import cv2
import time
#from pyhog import pyhog
import numpy as np
#from matplotlib import pyplot as plt
import selectivesearch
im = cv2.imread('00001.jpg')
start1 = time.time()

img_lbl, regions = selectivesearch.selective_search(im, scale=500, sigma=0.9, min_size=300)
print(len(regions))
j=0
while(j<len(regions)):
	cv2.rectangle(im, (regions[j]['rect'][0],regions[j]['rect'][1]), (regions[j]['rect'][0]+regions[j]['rect'][2],regions[j]['rect'][1]+regions[j]['rect'][3]),(255,0,0), 2)
	j=j+1

end1 = time.time()
print(end1-start1)
cv2.imshow('tracking_results',im)
c = cv2.waitKey(0)
#pic = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
#np.savetxt('fhog.txt',pic[:,:,2],fmt='%.4f')
#image = pic/255.
#hog_feature_t = pyhog.features_pedro(image, 4)
#print(hog_feature_t.shape)
#img_crop = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')



#zf = np.fft.fft2(img_crop, axes=(0, 1))
#print(zf[:,:,0].shape)
#np.savetxt('fhog1.txt',zf[:,:,0],fmt='%.4f')
#plt.matshow(np.real(zf[:,:,14]))
#plt.colorbar()
#plt.show()
#def test():
#    H = np.array([[3.16991321031250,52.4425641326457,2.73475152482102],[-8.76695007100685,43.4831885343255,-37.1705395356264],[-1.59218748085971,-24.3510937156625,12.8339630267640]])
#    U, S, V = np.linalg.svd(H) 
#    print(U)
#    print(V)
#    print(S)
 
#if __name__=='__main__':
#    test()