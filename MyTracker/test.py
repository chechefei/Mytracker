#-*- coding:utf-8 -*-
import cv2
import numpy as np
from sklearn.svm import SVC
from skimage import transform
import utils
import random

cv2.namedWindow('raw')
#cv2.namedWindow('gray')
cv2.namedWindow('rotate')
cv2.namedWindow('image_horizontal')
cv2.namedWindow('image_horizontal_result')
cv2.namedWindow('canny')
image = cv2.imread('./1371.jpg')
#从左上角开始，先纵向，后横向

print(image.shape)
#cv2.imshow('gray',image)
#image的索引方法是先y后x
image_patch = image[36:78,400:440,:]
tar_width = 50
tar_height = 50
tar_center = np.array([57,420])
#image[tar_center[0]:tar_center[0]+5,tar_center[1]:tar_center[1]+5,1]=255
#image[tar_center[0]:tar_center[0]+5,tar_center[1]:tar_center[1]+5,2]=0
#image[tar_center[0]:tar_center[0]+5,tar_center[1]:tar_center[1]+5,0]=0
cv2.imshow('raw',image)
print(image_patch.shape)
gray = cv2.cvtColor(image_patch,cv2.COLOR_BGR2GRAY)
edge = cv2.Canny(gray,50,150)
cv2.imshow('canny',edge)
a = 20
lines = cv2.HoughLinesP(edge, 1, np.pi / 18, a,
                        maxLineGap=1)
print(lines)
num_line = 0
theta_total = 0

for line in lines:
	x1, y1, x2, y2 = line[0]
	print(x1)
	print(y1)
	print(x2)
	print(y2)
	theta = np.arctan((y1-y2)/(x1-x2))
	num_line = num_line + 1
	theta_total = theta_total + theta
	print(theta_total)
	cv2.line(image_patch,(x1,y1),(x2,y2),(0,255,0),1)
theta_ave = theta_total / num_line
num = np.sin(theta_ave)*np.sin(theta_ave) - np.cos(theta_ave)*np.cos(theta_ave)
true_width = (tar_width * np.sin(theta_ave) - tar_height * np.cos(theta_ave))/num
true_height = (tar_height * np.sin(theta_ave) - tar_width * np.cos(theta_ave))/num
print('theta_ave')
print(theta_ave)
theta_delta = np.pi/2 + theta
if theta_ave > 0:
	theta_delta = - theta_delta
elif theta_ave <=0:
	theta_delta = theta_delta
print('theta_delta')
print(theta_delta)
theta2Horizontal =  theta_ave*180/np.pi
#cv2.resizeWindow("gray", 640, 480);
#cv2.imshow('gray',image_patch)
center = utils.Coord_trans([image.shape[0],image.shape[1]],tar_center,theta2Horizontal)
gray_rotate = transform.rotate(image,theta2Horizontal,resize = True)

print('image_shape')
print(image.shape)
#gray_rotate[center[0]:center[0]+5,center[1]:center[1]+5,1]=0
#gray_rotate[center[0]:center[0]+5,center[1]:center[1]+5,2]=255
#gray_rotate[center[0]:center[0]+5,center[1]:center[1]+5,0]=0
print('gray_rotate')
print(gray_rotate.shape)
cv2.imshow('rotate',gray_rotate)
image_horizontal = gray_rotate[center[0]-tar_width/2:center[0]+tar_width/2,center[1]-tar_height/2:center[1]+tar_height/2,:]
cv2.imshow('image_horizontal',image_horizontal)
sample_points = 40
sample_RGB = np.zeros((2*sample_points,3))
labels = np.zeros((2*sample_points,),dtype=np.int16)
labels[sample_points:sample_points*2,] = 1

coordinate = np.zeros((2,2*sample_points,),dtype=np.int16)
b = np.arange(sample_points)

list_x = np.arange(tar_width)
print(list_x)
index_y = np.reshape(random.sample(list_x,sample_points/2),(sample_points/2,))
print(index_y)
index_x = 1 + index_y % 2
coordinate[0,0:sample_points/2] = index_x
coordinate[1,0:sample_points/2] = index_y
index_x = tar_height - index_x
coordinate[0,sample_points/2:sample_points] = index_x
coordinate[1,sample_points/2:sample_points] = index_y
print(coordinate)
print(index_x)
list_x = np.arange(2,tar_width-2)
print(list_x)
index_y = np.reshape(random.sample(list_x,sample_points/2),(sample_points/2,))
print(index_y)
index_x = 1 + index_y % 2
center_xy = [tar_width/2,tar_height/2]
coordinate[0,sample_points:sample_points+sample_points/2] = center_xy[0] - index_x
coordinate[1,sample_points:sample_points+sample_points/2] = index_y
coordinate[0,sample_points+sample_points/2:sample_points*2] = center_xy[0] + index_x
coordinate[1,sample_points+sample_points/2:sample_points*2] = index_y
print(coordinate)
#image_horizontal[coordinate[0],coordinate[1],0] = 255
#image_horizontal[coordinate[0],coordinate[1],1] = 0
#image_horizontal[coordinate[0],coordinate[1],2] = 0
sample_RGB=image_horizontal[coordinate[0],coordinate[1],:]
clf = SVC()
print(type(image_horizontal))
clf.fit(sample_RGB,labels)
list_width = np.arange(tar_width)
list_height = np.arange(tar_height)
print(list_width)
for i in list_width:
	for j in list_height:
		if clf.predict([image_horizontal[i,j,:]])[0] == 1:
			image_horizontal[i,j,:] = image_horizontal[i,j,:]
		elif	clf.predict([image_horizontal[i,j,:]])[0] == 0:
			image_horizontal[i,j,:] = 0
print(sample_RGB.shape)
cv2.imshow('image_horizontal_result',image_horizontal)
cv2.waitKey(0)
