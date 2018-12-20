#-*- coding:utf-8 -*-
import cv2
#os模块进行常见的文件和目录操作
import os
import shutil
from Sequence import Sequence
from importlib import import_module

'''此文件完成跟踪流程的总体把控'''

def Tracking(Sequence, tracker_list, visualize = False):

    if not os.path.exists('results/'):
        #创建目录
        os.mkdir("results")

    print 'generate images.txt and region.txt files...'

    with open("images.txt","w") as f:
        while Sequence._frame < len(Sequence._images):
            #将多个文件的路径写进txt文档
            f.write(Sequence.frame()+'\n')
            Sequence._frame+=1
        Sequence._frame = 0
    with open("region.txt", "w") as f:
        f.write(open(os.path.join(Sequence.seqdir, 'groundtruth.txt'), 'r').readline())

    print 'start tracking...'

    for str in tracker_list:
        print 'tracking using: '+str
        print(str)
        import_module(str)
        if not os.path.exists('results/'+str+'/'+Sequence.name):
            os.makedirs('results/'+str+'/'+Sequence.name)
        shutil.move("output.txt", 'results/'+str+'/'+Sequence.name+'/output.txt')
    #进行文件的删除
    os.remove("images.txt")
    os.remove("region.txt")

    print 'Done!!'

#sequence = Sequence(path='/home/chefei/Documents/UAV-benchmark-S', name='Trunk', region_format='rectangle')
sequence = Sequence(path='/home/chefei/Documents/UAV-benchmark-S', name='double', region_format='rectangle')
Tracking(sequence,tracker_list=['KCFtracker'],visualize=True)