# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import cv2

img = cv2.imread('111.png')
print(img.shape)
x,y,z = img[0],img[1],img[2]
ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
num = img.shape[0]*img.shape[1]
#  将数据点分成三部分画，在颜色上有区分度
ax.scatter(x[:num], y[:num], z[:num], c='y')  # 绘制数据点


ax.set_zlabel('Z')  # 坐标轴
ax.set_ylabel('Y')
ax.set_xlabel('X')
plt.show()