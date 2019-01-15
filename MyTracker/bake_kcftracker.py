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
import urllib
from pyheatmap.heatmap import HeatMap

class KCFtracker:
    def __init__(self, image, region):
        #目标的大小
        self.target_size = np.array([region.height, region.width])
        s = max(region.height, region.width)
        #目标的位置
        self.pos = [region.y + region.height / 2, region.x + region.width / 2]
        #视情况看是不是要缩小图像，太大了计算量很大，不划算
        self.resize_image = (np.sqrt(np.prod(self.target_size)) >= 100)
        if self.resize_image:
            self.pos = np.floor(self.pos/2)
            self.target_size = np.floor(self.target_size/2)
        #特征序列，用于确定需要采用的特征
        self.feature_list = np.array(['fhog','',''])
        #颜色对应矩阵，用于cn特征的提取
        self.w2c = np.load('w2crs.npy')
        #特征通道数的计算，用于初始化特征矩阵
        self.num_feature_ch = 0
        if 'cn' in self.feature_list:
            self.num_feature_ch = self.num_feature_ch + 10
        if 'fhog' in self.feature_list:
            self.num_feature_ch = self.num_feature_ch + 18
        if 'gray' in self.feature_list:
            self.num_feature_ch = self.num_feature_ch + 1

        #padding值，搜索区域
        padding = 5
        select_padding = np.int(np.ceil(padding*np.sqrt(2)))
        self.select_patch_size = np.floor(np.array([s,s]) * (1+select_padding))
        #self.select_pos = np.ceil(self.select_patch_size/2)
        #提取较大区域的图像，用于旋转生成
        img4sample = utils.get_subwindow(image, self.pos, self.select_patch_size)
        if 'fhog' in self.feature_list:
            self.cell_size = np.int(np.round(s/15));
        else:
            self.cell_size = 1
        
        
        #self.patch_size = np.floor(np.array((s* (1+padding),s* (1+padding))))
        self.patch_size = np.floor(self.target_size * (1 + padding))
        self.patch_size_cell = np.round(self.patch_size/self.cell_size)
        #用于存储滤波器序列，sample序列和sample傅里叶变换序列，用于后续查表
        self.filter_sequence1 = np.zeros((24,np.int((self.patch_size_cell[0])),np.int(self.patch_size_cell[1])))
        self.x_sequence = np.zeros((24,np.int(self.patch_size_cell[0]),np.int(self.patch_size_cell[1]),self.num_feature_ch))
        self.xf_sequence1 = np.zeros((24,np.int(self.patch_size_cell[0]),np.int(self.patch_size_cell[1]),self.num_feature_ch))
        
        self.filter_sequence = self.filter_sequence1.astype(np.complex128)
        self.xf_sequence = self.xf_sequence1.astype(np.complex128)
        #self.angle_index = np.int(np.round(-theta2Horizontal/15))
        #if self.angle_index < 0:
        #    self.angle_index = angle_index + 24
        self.angle_index = 0
        spatial_bandwidth_sigma_factor = 1 / float(16)
        output_sigma = np.sqrt(np.prod(self.target_size)) * spatial_bandwidth_sigma_factor/self.cell_size
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
            print('img_crop')
            print(img_crop.shape)
            print(self.num_feature_ch)
            #hog_feature_t = pyhog.features_pedro(img_crop / 255., self.cell_size)
            #print('hog_feature_t')
            #print(hog_feature_t.shape)
            #img_crop = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
            #img_crop = img_crop[:,:,0:18]
            img_cro = utils.get_feature_map(img_crop,self.feature_list,self.num_feature_ch,
                self.patch_size_cell,self.w2c,self.cell_size)
            #print('img_crop')
            #print(img_crop.shape)
            self.x = np.multiply(img_cro, self.cos_window[:, :, None])
            #print("加余弦窗后的大小："+str(self.x.shape[:3]))
            self.xf = np.fft.fft2(self.x, axes=(0, 1))
            #print("傅里叶变换后的大小："+str(self.xf.shape[:3]))
            self.feature_bandwidth_sigma = 0.2
            k = utils.dense_gauss_kernel(self.feature_bandwidth_sigma, self.xf, self.x)
            lambda_value = 1e-4
            self.alphaf = np.divide(yf, np.fft.fft2(k, axes=(0, 1)) + lambda_value)
            self.filter_sequence[i,:,:] = self.alphaf
            #np.savetxt('filter_sequence.txt',self.filter_sequence[i,:,:],fmt='%.4f')
            #np.savetxt('alphaf.txt',self.alphaf,fmt='%.4f')
            self.x_sequence[i,:,:,:] = self.x
            self.xf_sequence[i,:,:,:] = self.xf
            #print('alphaf')
        end2 = time.time()
        print ('生成图像耗时：'+str(end2-start2))
        print(self.filter_sequence.shape)
        print(self.alphaf.shape)
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

        #hog_feature_t = pyhog.features_pedro(test_crop / 255., self.cell_size)
        #hog_feature_t = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
        #hog_feature_t = hog_feature_t[:,:,0:18]
        hog_feature_t = utils.get_feature_map(test_crop,self.feature_list,self.num_feature_ch,self.patch_size_cell,self.w2c,self.cell_size)

        z = np.multiply(hog_feature_t, self.cos_window[:, :, None])
        print(z.shape)
        zf = np.fft.fft2(z, axes=(0, 1))
        angle_index_series = (np.array((self.angle_index-1,self.angle_index,self.angle_index+1))+24)%24
        response_map_series = np.zeros((24,np.int(self.patch_size_cell[0]),np.int(self.patch_size_cell[1])))
        j = 0

        end3 = time.time()
        print ('生成检测用时：'+str(end3-start3))
        start4 = time.time()
    
        for i in angle_index_series:
            k_test = utils.dense_gauss_kernel(self.feature_bandwidth_sigma,self.xf_sequence[i,:,:,:],self.x_sequence[i,:,:,:],zf,z)
            kf_test = np.fft.fft2(k_test, axes=(0, 1))
            alphaf_test = self.filter_sequence[i,:,:]
            response = np.real(np.fft.ifft2(np.multiply(alphaf_test, kf_test)))
            response_map_series[i,:,:] = response
            
            #plt.imshow(response, extent=[0, 1, 0, 1])
            self.response_series[j] = np.max(response)
            self.v_centre[j], self.h_centre[j] = np.unravel_index(response.argmax(), response.shape)
            j = j + 1
        print('response_series')

        max_response_index = np.where(self.response_series==np.max(self.response_series))[0][0]
        print(self.response_series)
        v = self.v_centre[max_response_index]
        h = self.h_centre[max_response_index]
        self.angle_index = angle_index_series[max_response_index]
        response4show = np.reshape(response_map_series[self.angle_index,:,:],
            (np.int(self.patch_size_cell[0]),np.int(self.patch_size_cell[1])))
        cv2.imshow('response',response4show)
        #plt.matshow(response4show)
        #plt.colorbar()
        #plt.show()
        #plt.pause(0.033)
        
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

