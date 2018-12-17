#-*- coding:utf-8 -*-
import sys
import cv2
import vot
import numpy as np
import utils
import Sequence
import matplotlib.pyplot as plt
import time
import collections
import matplotlib.patches as patches
from pyhog import pyhog

class padding:
    def __init__(self):
        self.generic = 1.8
        self.large = 1
        self.height = 0.4


class Rtracker:
    
    def __init__(self, image, region):
        #调整位置滤波器输出的参数
        output_sigma_factor = 1 / float(16)
        self.lamda = 1e-2
        self.interp_factor = 0.025
        
        #目标尺寸
        self.target_size = np.array([region.height, region.width])
        #目标中心位置
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        #初始目标大小
        init_target_size = self.target_size
        #基本目标尺寸，是根据目标尺寸和当前的尺度变化因子决定的
        self.base_target_size = self.target_size
        #image.shape[:2]是返回图像（矩阵）前两维的大小，如果是1就返回第一维
        #省略就返回三个维度的大小
        #此步骤返回padding区域的大小，另外涉及到了用类初始化参数并作为形参用于另一函数的方法
        self.sz = utils.get_window_size(self.target_size, image.shape[:2],padding())
        #位置滤波器和尺度滤波器的参数，理想的高斯响应里面的sigma
        output_sigma = np.sqrt(np.prod(self.target_size)) * output_sigma_factor
        #scale_sigma = np.sqrt(nScales) * scale_sigma_factor
        #通过在x,y方向生成网格，从而生成相应的理想高斯响应
        #arange(n)生成一个序列0到n，如果是两个参数就是从m到n，如果是三个参数，第三个参数就是步长
        grid_y = np.arange(np.floor(self.sz[0])) - np.floor(self.sz[0] / 2)
        grid_x = np.arange(np.floor(self.sz[1])) - np.floor(self.sz[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))

        #求位置高斯响应的傅里叶变换
        self.yf = np.fft.fft2(y, axes=(0, 1))

        #获取特征图,参数为目标中心位置和尺寸，特征为hog
        feature_map = utils.get_subwindow(image, self.pos, self.sz, feature='hog')
        #对特征图使用余弦窗，并进行傅里叶变换
        self.cos_window = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
        x_hog = np.multiply(feature_map, self.cos_window[:, :, None])
        xf = np.fft.fft2(x_hog, axes=(0, 1))
        self.x_num = np.multiply(self.yf[:, :, None], np.conj(xf))
        self.x_den = np.sum(np.multiply(xf, np.conj(xf)), axis=2)
    
    def track(self, image):
        # ---------------------------------------track--------------------------------- #
        test_patch = utils.get_subwindow(image, self.pos, self.sz)
        
        hog_feature_t = pyhog.features_pedro(test_patch / 255., 1)
        hog_feature_t = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
        xt = np.multiply(hog_feature_t, self.cos_window[:, :, None])
        xtf = np.fft.fft2(xt, axes=(0, 1))
        #计算响应，直接多通道叠加
        response = np.real(np.fft.ifft2(np.divide(np.sum(np.multiply(self.x_num, xtf),axis=2),
                                                        (self.x_den + self.lamda))))
        #找响应最大值
        v_centre, h_centre = np.unravel_index(response.argmax(), response.shape)
        vert_delta, horiz_delta = \
            [(v_centre - response.shape[0] / 2),
             (h_centre - response.shape[1] / 2)]
        #新的位置
        self.pos = [self.pos[0] + vert_delta, self.pos[1] + horiz_delta]

        
        # ---------------------------------------update--------------------------------- #
        update_patch = utils.get_subwindow(image, self.pos, self.sz)
        hog_feature_l = pyhog.features_pedro(update_patch / 255., 1)
        hog_feature_l = np.lib.pad(hog_feature_l, ((1, 1), (1, 1), (0, 0)), 'edge')
        xl = np.multiply(hog_feature_l, self.cos_window[:, :, None])
        xlf = np.fft.fft2(xl, axes=(0, 1))
        #更新位置滤波器
        new_x_num = np.multiply(self.yf[:, :, None], np.conj(xlf))
        new_x_den = np.real(np.sum(np.multiply(xlf, np.conj(xlf)),axis=2))

        #滤波器学习
        self.x_num = (1 - self.interp_factor) * self.x_num + self.interp_factor * new_x_num
        self.x_den = (1 - self.interp_factor) * self.x_den + self.interp_factor * new_x_den
        self.target_size = self.base_target_size

        return vot.Rectangle(self.pos[1] - self.target_size[1] / 2,
                             self.pos[0] - self.target_size[0] / 2,
                             self.target_size[1],
                             self.target_size[0]
                             )
        
#定义了VOT接口的函数句柄
handle = vot.VOT("rectangle")
#返回第一帧目标的位置
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)
cv2.namedWindow('tracking_results')
image = cv2.imread(imagefile)
tracker = Rtracker(image, selection)

while True:
    print('--------------------------跟踪一帧---------------------------------')
    frame_index = handle.return_frame_index()
    #返回图像路径
    imagefile = handle.frame()
    if not imagefile:
        break
    #读取图像
    image = cv2.imread(imagefile)
    #利用图像进行跟踪，并返回跟踪结果
    start = time.time()
    region = tracker.track(image)
    end = time.time()
    print('跟踪时间：')
    print end-start
    print(region)
    cv2.rectangle(image, (int(region.x),int(region.y)), (int(region.x+region.width),int(region.y+region.height)),(255,0,0), 2)
    cv2.putText(image, 'fps: ' + str(round(1/(end-start),1)), (20,60), 1, 2, (0,0,255), 2)  
    cv2.putText(image, 'frame: ' + str(frame_index), (20,30), 1, 2, (0,0,255), 2) 
    cv2.imshow('tracking_results',image)
    inteval = 10;
    c = cv2.waitKey(inteval) & 0xFF
    if c==27 or c==ord('q'):
        cv2.destroyAllWindows()
        break
    #此语句完成了跟踪结果的添加和帧序号的叠加
    handle.report(region)
cv2.destroyAllWindows()
handle.quit()


