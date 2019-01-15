#-*- coding:utf-8 -*-
"""
\file Sequence.py

@author Xiaofeng Mao

@date 2017.9.27

"""

import sys
import copy
import collections
import os

#namedtuple是一个函数，用来自定义tuple函数，并且规定了tuple元素的个数，并且使用属性而不是索引来引用tuple的某个元素
#记住产生tuple的方法，并且赋值方法
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])
Point = collections.namedtuple('Point', ['x', 'y'])
Polygon = collections.namedtuple('Polygon', ['points'])

#解析区域
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

#改变boundbox的给出形式
def convert_region(region, to):

    if to == 'rectangle':
        #isinstance用来判断对象是否是一个已知的类型
        if isinstance(region, Rectangle):
            return copy.copy(region)
        elif isinstance(region, Polygon):
            #查看计算机中float类型的最大最小值，防止溢出
            top = sys.float_info.max
            bottom = sys.float_info.min
            left = sys.float_info.max
            right = sys.float_info.min

            for point in region.points: 
                #分别求取最小最大x，y值，换算得到长宽
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

class Sequence(object):
    """ Base class for Python VOT integration """
    #sequence = Sequence(path='/home/chefei/Documents/UAV-benchmark-S', name='S0001', region_format='rectangle')
    def __init__(self, path, name, region_format = 'rectangle'):
        #获取序列名称
        self.name = name
        """ Constructor
        
        Args: 
            region_format: Region format options
        """
        #断言，查看region_format是否是规定的形式之内的，否则返回错误
        assert(region_format in ['rectangle', 'polygon'])

        #如果没有指定序列名称，那么序列路径为文件路径
        if len(name) == 0:
            self.seqdir = path
        else:
            #如果有指定序列名称，那么序列路径进行合成
            #join表示连接目录与文件名
            self.seqdir = os.path.join(path, name)

        flag = False
        self._images=[]
        #向图像列表中添加图像文件
        for _, _, files in os.walk(self.seqdir):
            for file in files:
                if file.endswith('jpg') or file.endswith('png'):
                    self._images.append(file)
                    
        #sort是列表内置的一个排序算法，直接修改原列表
        #进行排序，程序的意思是将_images列表里面的文件名的后四个字符(.jpg或.png)去掉
        #并把剩下的字符转为int形式作为排序的key进行排序
        if 'img' in self._images[0]:
            self._images.sort(key= lambda x:int(x[3:-4]))
        else:
            self._images.sort(key= lambda x:int(x[0:-4]))
        self.groundtruth = []
        #readlines用于读取所有行，直到遇到结束符EOF
        #读取了所有帧的groundtruth，用于和result进行比对，进行精确性分析
        for x in open(os.path.join(self.seqdir, 'groundtruth.txt'), 'r').readlines():
            self.groundtruth.append(convert_region(parse_region(x), region_format))

        self._frame = 0
        #readline 每次只读一行，用在初始化中，即只读取第一行
        self._region = convert_region(parse_region(open(os.path.join(self.seqdir, 'groundtruth.txt'), 'r').readline()), region_format)
        self._result = []
        self._region_format = region_format
    
    def region(self):
        """
        Send configuration message to the client and receive the initialization 
        region and the path of the first image 
        
        Returns:
            initialization region 
        """          

        return self._region

    def report(self, region):
        """
        Report the tracking results to the client
        
        Arguments:
            region: region for the frame    
        """
        #断言，查看返回的region格式是否正确
        assert(isinstance(region, Rectangle) or isinstance(region, Polygon))

        self._result.append(region)
        self._frame += 1
        
    def frame(self):
        """
        Get a frame (image path) from client 
        
        Returns:
            absolute path of the image
        """
        #如果帧序号已经超出图片的总数，那么不返回
        if self._frame >= len(self._images):
            return None
        #因为已经进行了sort，序号和图像中的序号是匹配的，这里就要求图像序列中不存在断层的情况
        return os.path.join(self.seqdir, self._images[self._frame])
    def quit(self):
        if hasattr(self, '_result'):
            #跟踪结束，输出跟踪结果到output.txt文件中
            with open('output.txt', 'w') as f:
                for r in self._result:
                    f.write(encode_region(r))
                    f.write('\n')