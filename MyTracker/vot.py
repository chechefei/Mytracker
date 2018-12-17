#-*- coding:utf-8 -*-
"""
\file vot.py

@brief Python utility functions for VOT integration

@author Luka Cehovin, Alessio Dore

@date 2016

"""

import sys
import copy
import collections


Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

def parse_region(string):
    tokens = map(float, string.split(','))
    if len(tokens) == 4:
        return Rectangle(tokens[0], tokens[1], tokens[2], tokens[3])
    elif len(tokens) % 2 == 0 and len(tokens) > 4:
        return Polygon([Point(tokens[i],tokens[i+1]) for i in xrange(0,len(tokens),2)])
    return None

def encode_region(region):
    if isinstance(region, Polygon):
        return ','.join(['{},{}'.format(p.x,p.y) for p in region.points])
    elif isinstance(region, Rectangle):
        return '{},{},{},{}'.format(region.x, region.y, region.width, region.height)
    else:
        return ""

def convert_region(region, to):

    if to == 'rectangle':

        if isinstance(region, Rectangle):
            return copy.copy(region)
        elif isinstance(region, Polygon):
            top = sys.float_info.max
            bottom = sys.float_info.min
            left = sys.float_info.max
            right = sys.float_info.min

            for point in region.points: 
                top = min(top, point.y)
                bottom = max(bottom, point.y)
                left = min(left, point.x)
                right = max(right, point.x)

            return Rectangle(left, top, right - left, bottom - top)

        else:
            return None  
    if to == 'polygon':

        if isinstance(region, Rectangle):
            points = []
            points.append((region.x, region.y))
            points.append((region.x + region.width, region.y))
            points.append((region.x + region.width, region.y + region.height))
            points.append((region.x, region.y + region.height))
            return Polygon(points)

        elif isinstance(region, Polygon):
            return copy.copy(region)
        else:
            return None  

    return None

class VOT(object):
    #初始化函数，后面的函数会对初始化后的参数进行操作
    def __init__(self, region_format):
        """ 

        """
        assert(region_format in ['rectangle', 'polygon'])
        #该语句完成了所有文件路径的读入，读入后_files是一个所有图片序列的列表
        self._files = [x.strip('\n') for x in open('images.txt', 'r').readlines()]
        self._frame = 0
        #读入第一帧中目标的位置
        self._region = convert_region(parse_region(open('region.txt', 'r').readline()), region_format)
        self._result = []
        #print(self._region)
        
    def region(self):
        """
        返回目标区域 
        """          
        return self._region
    def return_frame_index(self):
        return self._frame

    def report(self, region):
        """
        Report the tracking results to the client
        
        Arguments:
            region: region for the frame    
        """
        assert(isinstance(region, Rectangle) or isinstance(region, Polygon))
        self._result.append(region)
        self._frame += 1
        
    def frame(self):
        """
        Get a frame (image path) from client 
        
        Returns:
            absolute path of the image
        """
        
            #如果帧的序号大于图像序列的长度，不返回
        if self._frame >= len(self._files):
            return None
            #如果还有图像序列，那么返回相应序号的图像
        return self._files[self._frame]

    def quit(self):
        if hasattr(self, '_result'):
            #跟踪结束，输出跟踪结果到output.txt文件中
            with open('output.txt', 'w') as f:
                for r in self._result:
                    f.write(encode_region(r))
                    f.write('\n')


