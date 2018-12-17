#-*- coding:utf-8 -*-
import sys
import cv2
import numpy as np
import utils
import vot
import matplotlib.pyplot as plt
import time
import collections
from skimage import transform
import matplotlib.patches as patches
from pyhog import pyhog
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class KCFtracker:
    def __init__(self, image, region):
        #目标的大小
        self.target_size = np.array([region.height, region.width])
        #长和宽中的较大者，用于确定判定为直线的像素长度参考
        s = max(region.height, region.width)
        #目标的位置
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]

        #获取目标区域的图像，用于检测目标朝向，方便下一步修正为水平并生成旋转采样样本
        #target_patch = utils.get_subwindow(image, self.pos, self.target_size)
        #进行灰度转换，canny边缘检测和概率hough直线检测，需要注意的是，此时的图像已经是经过归一化的，需要还原为255图像
        #gray = cv2.cvtColor(np.uint8(target_patch*255.),cv2.COLOR_BGR2GRAY)
        #edge = cv2.Canny(gray,50,150)
        #直线检测，确定车辆大致方向
        #line_threshold = int(np.floor(s/2))
        #num_line = 0
        #theta_total = 0
        #lines = cv2.HoughLinesP(edge, 1, np.pi / 18, line_threshold,maxLineGap=1)
        #for line in lines:
        #    x1, y1, x2, y2 = line[0]
        #    theta = np.arctan((y1-y2)/(x1-x2))
        #    num_line = num_line + 1
        #    theta_total = theta_total + theta
        #通过图像的长宽和目标的方向角度求出目标的长宽，方便在旋转后获取精确的目标图像
        #theta_ave = theta_total / num_line
        #print(theta_ave)

        #theta2Horizontal = theta_ave*180/np.pi
        #进行图像旋转到目标水平

        #padding值，搜索区域
        padding = 5
        select_padding = np.int(np.ceil(padding*np.sqrt(2)))
        self.select_patch_size = np.floor(np.array([s,s]) * (1 + select_padding))
        self.select_pos = np.ceil(self.select_patch_size/2)
        img4sample = utils.get_subwindow(image, self.pos, self.select_patch_size)
        #center = utils.Coord_trans([img4sample.shape[0],img4sample.shape[1]],self.select_pos,theta2Horizontal)
        #img4sample_rotate = transform.rotate(img4sample,theta2Horizontal,resize = True)
        
        self.cell_size = np.int(np.round(s/15));
        #self.cell_size = 4
        
        self.patch_size = np.floor(np.array((s*(1 + padding),s*(1 + padding))))
        self.patch_size_cell = np.round(self.patch_size/self.cell_size)
        #用于存储滤波器序列，sample序列和sample傅里叶变换序列，用于后续查表
        self.filter_sequence = np.zeros((24,np.int((self.patch_size_cell[0])),np.int(self.patch_size_cell[1])))
        self.x_sequence = np.zeros((24,np.int((self.patch_size_cell[0])),np.int(self.patch_size_cell[1]),31))
        self.xf_sequence = np.zeros((24,np.int((self.patch_size_cell[0])),np.int(self.patch_size_cell[1]),31))
        #self.angle_index = np.int(np.round(-theta2Horizontal/15))
        #if self.angle_index < 0:
        #    self.angle_index = angle_index + 24
        self.angle_index = 0
        spatial_bandwidth_sigma_factor = 1 / float(16)
        output_sigma = np.sqrt(np.prod(self.target_size)) * spatial_bandwidth_sigma_factor
        grid_y = np.arange(np.floor(self.patch_size_cell[0])) - np.floor(self.patch_size_cell[0] / 2)
        grid_x = np.arange(np.floor(self.patch_size_cell[1])) - np.floor(self.patch_size_cell[1] / 2)
        rs, cs = np.meshgrid(grid_x, grid_y)
        y = np.exp(-0.5 / output_sigma ** 2 * (rs ** 2 + cs ** 2))
        self.cos_window = np.outer(np.hanning(y.shape[0]), np.hanning(y.shape[1]))
        yf = np.fft.fft2(y, axes=(0, 1))

        start2 = time.time()


        index = np.arange(24)
        for i in index:
            angle = i*15
            #
            sample_image = transform.rotate(img4sample,angle,resize = True)
            sample_center = np.floor(np.array((sample_image.shape[0],sample_image.shape[1]))/2)
            img_crop = utils.get_subwindow(sample_image, sample_center, self.patch_size)
            #if i == 5:
            #    print(sample_image.shape)
            #    print(sample_center)
            #    cv2.imshow('response',sample_image)
            #    cv2.imshow('sample_image',img_crop)
            #    print(img_crop.shape)
            #
            print('img_crop')
            print(img_crop.shape)
            hog_feature_t = pyhog.features_pedro(img_crop / 255., self.cell_size)
            print('hog_feature_t')
            print(hog_feature_t.shape)
            img_crop = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
            print('img_crop')
            print(img_crop.shape)
            self.x = np.multiply(img_crop, self.cos_window[:, :, None])
            #print("加余弦窗后的大小："+str(self.x.shape[:3]))
            self.xf = np.fft.fft2(self.x, axes=(0, 1))
            #print("傅里叶变换后的大小："+str(self.xf.shape[:3]))
            self.feature_bandwidth_sigma = 0.2
            k = utils.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
            lambda_value = 1e-4
            self.alphaf = np.divide(yf, np.fft.fft2(k, axes=(0, 1)) + lambda_value)
            self.filter_sequence[i,:,:] = self.alphaf
            self.x_sequence[i,:,:,:] = self.x
            self.xf_sequence[i,:,:,:] = self.xf
            #print('alphaf')
        end2 = time.time()
        print ('生成图像耗时：'+str(end2-start2))

        print(self.filter_sequence.shape)
        #cv2.waitKey(0)
        #print(self.filter_sequence)
        self.response_series = np.array([0.,0.,0.])
        self.v_centre = np.array([0.,0.,0.])
        self.h_centre = np.array([0.,0.,0.])
    def track(self, image):
        print(self.cell_size)
        start3 = time.time()
    

        test_crop = utils.get_subwindow(image, self.pos, self.patch_size)
        cv2.imshow('hahaha',test_crop)
        hog_feature_t = pyhog.features_pedro(test_crop / 255., self.cell_size)
        hog_feature_t = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
        z = np.multiply(hog_feature_t, self.cos_window[:, :, None])
        print(z.shape)
        zf = np.fft.fft2(z, axes=(0, 1))
        angle_index_series = (np.array((self.angle_index-1,self.angle_index,self.angle_index+1))+24)%24
        
        j = 0

        end3 = time.time()
        print ('生成检测用时：'+str(end3-start3))
        start4 = time.time()
    


        for i in angle_index_series:
            k_test = utils.dense_gauss_kernel(self.feature_bandwidth_sigma,self.xf_sequence[i,:,:,:],self.x_sequence[i,:,:,:],zf,z)
            kf_test = np.fft.fft2(k_test, axes=(0, 1))
            alphaf_test = self.filter_sequence[i,:,:]
            response = np.real(np.fft.ifft2(np.multiply(alphaf_test, kf_test)))
            cv2.imshow('response',response)
            self.response_series[j] = np.max(response)
            self.v_centre[j], self.h_centre[j] = np.unravel_index(response.argmax(), response.shape)
            j = j + 1
        print('response_series')
        max_response_index = np.where(self.response_series==np.max(self.response_series))[0][0]
        print(self.response_series)
        v = self.v_centre[max_response_index]
        h = self.h_centre[max_response_index]
        print(self.angle_index)
        print(self.angle_index)
        print(self.angle_index)
        print(self.angle_index)
        print(self.angle_index)
        self.angle_index = angle_index_series[max_response_index]
        end4 = time.time()
        print ('三个滤波器用时：'+str(end4-start4))
        vert_delta, horiz_delta = [v - response.shape[0] / 2,h - response.shape[1] / 2]
        self.pos = [self.pos[0] + vert_delta*self.cell_size, self.pos[1] + horiz_delta*self.cell_size]
       
        return vot.Rectangle(self.pos[1] - self.target_size[1] / 2,
                             self.pos[0] - self.target_size[0] / 2,
                             self.target_size[1],
                             self.target_size[0]
                            )
        
handle = vot.VOT("rectangle")
selection = handle.region()

imagefile = handle.frame()
if not imagefile:
    sys.exit(0)

image = cv2.imread(imagefile)/255.

print('go here 1')
#cv2.namedWindow('response')
#cv2.namedWindow('sample_image')
tracker = KCFtracker(image, selection)
while True:
    frame_index = handle.return_frame_index()
    print('--------------------------跟踪第'+str(frame_index)+'帧---------------------------------')
    imagefile = handle.frame()
    if not imagefile:
        break
    image = cv2.imread(imagefile)/255.
    #利用图像进行跟踪，并返回跟踪结果
    start1 = time.time()
    region = tracker.track(image)
    end1 = time.time()
    print ('帧率为：'+str(int(1/(end1-start1))))
    print(region)
    cv2.rectangle(image, (int(region.x),int(region.y)), (int(region.x+region.width),int(region.y+region.height)),(255,0,0), 2)
    cv2.putText(image, 'fps: ' + str(round(1/(end1-start1),1)), (20,60), 1, 2, (0,0,255), 2)  
    cv2.putText(image, 'frame: ' + str(frame_index), (20,30), 1, 2, (0,0,255), 2) 
    cv2.imshow('tracking_results',image)
    inteval = 30;
    c = cv2.waitKey(inteval) & 0xFF
    if c==27 or c==ord('q'):
        cv2.destroyAllWindows()
        break
    #此语句完成了跟踪结果的添加和帧序号的叠加
    handle.report(region)
    
handle.quit()

