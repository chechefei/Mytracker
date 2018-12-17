#-*- coding:utf-8 -*-
import cv2
import numpy as np
from skimage import transform

cv2.namedWindow('gray')
cv2.namedWindow('rotate')
image = cv2.imread('./00000002.jpg')
print(image.shape)
#cv2.imshow('gray',image)
#image的索引方法是先y后x
image_patch = image[256:285,519:546,:]
print(image_patch.shape)
gray = cv2.cvtColor(image_patch,cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray,50,150)
a = 10
lines = cv2.HoughLines(edge,1,np.pi/18,a)
print(lines)
num_line = 0
theta_total = 0
for line in lines:
	rho,theta = line[0]
	a = np.cos(theta)
	b = np.sin(theta)
	x0 = a*rho
	y0 = b*rho
	x1 = int(x0 + 1000*(-b))
	y1 = int(y0 + 1000*(a))
	x2 = int(x0 - 1000*(-b))
	y2 = int(y0 - 1000*(a))
	cv2.line(image_patch,(x1,y1),(x2,y2),(0,0,255),1)
	num_line = num_line + 1
	theta_total = theta_total + theta
	print(theta_total)
theta_ave = theta_total / num_line
print(theta_ave)
theta2Horizontal = - theta_ave*180/np.pi
cv2.resizeWindow("gray", 640, 480);
cv2.imshow('gray',image_patch)
gray_rotate = transform.rotate(gray,theta2Horizontal,resize = True)
cv2.imshow('rotate',gray_rotate)
cv2.waitKey(0)
