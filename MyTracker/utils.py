#-*- coding:utf-8 -*-
import numpy as np
from scipy import misc
from skimage import transform
import cv2
from pyhog import pyhog


def get_window_size(target_sz, im_sz, padding):

    if (target_sz[0] / target_sz[1] > 2):
        #如果是高明显大于宽的，限制一下，高方向搜索区域0.4，宽方向搜索区域1.8
        # For objects with large height, we restrict the search window with padding.height
        window_sz = np.floor(np.multiply(target_sz, [1 + padding.height, 1 + padding.generic]))

    elif np.prod(target_sz)/np.prod(im_sz) > 0.05:
        #对于大尺寸的目标，扩展区域1
        # For objects with large height and width and accounting for at least 10 percent of the whole image,
        # we only search 2xheight and width
        window_sz = np.floor(target_sz * (1 + padding.large))

    else:
        #普通目标搜索区域1.8
        window_sz = np.floor(target_sz * (1 + padding.generic))

    return window_sz




def get_subwindow(im, pos, sz, scale_factor = None, feature='raw'):
    """
    Obtain sub-window from image, with replication-padding.
    Returns sub-window of image IM centered at POS ([y, x] coordinates),
    with size SZ ([height, width]). If any pixels are outside of the image,
    they will replicate the values at the borders.

    The subwindow is also normalized to range -0.5 .. 0.5, and the given
    cosine window COS_WINDOW is applied
    (though this part could be omitted to make the function more general).
    """
    #测试sz是不是标量，如果是，就是方形窗口
    if np.isscalar(sz):  # square sub-window
        sz = [sz, sz]

    sz_ori = sz
    if scale_factor != None:
        sz = np.floor(sz*scale_factor)
    #获取有效区域位置的坐标序号，注意数列/矩阵的运用
    ys = np.floor(pos[0]) + np.arange(sz[0], dtype=int) - np.floor(sz[0] / 2)
    xs = np.floor(pos[1]) + np.arange(sz[1], dtype=int) - np.floor(sz[1] / 2)

    ys = ys.astype(int)
    xs = xs.astype(int)

    # check for out-of-bounds coordinates and set them to the values at the borders
    # 有可能这个取特征的区域已经超出了图像的范围，将超出的范围置为边界数值
    ys[ys < 0] = 0
    ys[ys >= im.shape[0]] = im.shape[0] - 1

    xs[xs < 0] = 0
    xs[xs >= im.shape[1]] = im.shape[1] - 1

    out = im[np.ix_(ys, xs)]


    #misc.imresize函数实现图像的大小调整，可以给数字对表示调整到固定大小，小于1的数字表示按照比例调整，大于1就按照百分比调整
    if scale_factor != None:
        out = misc.imresize(out, sz_ori.astype(int))
        
    if feature == 'hog':
        from pyhog import pyhog
        hog_feature = pyhog.features_pedro(out / 255., 1)
        out = np.lib.pad(hog_feature, ((1, 1), (1, 1), (0, 0)), 'edge')
        print('getting hog features in get_subwindow')
    return out

def merge_features(features):
    num, h, w = features.shape
    row = int(np.sqrt(num))
    merged = np.zeros([row * h, row * w])

    for idx, s in enumerate(features):
        i = idx // row
        j = idx % row
        merged[i * h:(i + 1) * h, j * w:(j + 1) * w] = s


    return merged

def dense_gauss_kernel(sigma, xf, x, zf = None, z = None):
    
    N = xf.shape[0] * xf.shape[1]
    xx = np.dot(x.flatten().transpose(), x.flatten())  # squared norm of x
    
    if zf is None:
        # auto-correlation of x
        zf = xf
        zz = xx
    else:
        zz = np.dot(z.flatten().transpose(), z.flatten())  # squared norm of y
    xyf = np.multiply(zf, np.conj(xf))
    #如果是多通道的特征
    if len(xyf.shape) == 3:
        xyf_ifft = np.fft.ifft2(np.sum(xyf, axis=2))
        #xyf_ifft = np.fft.ifft2(xyf)
    elif len(xyf.shape) == 2:
        xyf_ifft = np.fft.ifft2(xyf)

    c = np.real(xyf_ifft)
    d = np.real(xx) + np.real(zz) - 2 * c
    k = np.exp(-1. / sigma ** 2 * np.abs(d) / N)

    return k

def get_scale_subwindow(im,pos,base_target_size, scaleFactors,
                        scale_window, scale_model_sz):
    from pyhog import pyhog
    nScales = len(scaleFactors)
    out = []
    #每个尺度都进行hog采样，都要resize到同样的尺度
    for i in range(nScales):
        patch_sz = np.floor(base_target_size * scaleFactors[i])
        scale_patch = get_subwindow(im, pos, patch_sz)
        #resize改变图片的大小，rescale图片按照比例缩放，rotate进行图像旋转
        im_patch_resized = transform.resize(scale_patch, scale_model_sz,mode='reflect')
        temp_hog = pyhog.features_pedro(im_patch_resized, 4)
        out.append(np.multiply(temp_hog.flatten(), scale_window[i]))

    return np.asarray(out)


def Coord_trans(image_size,tar_center,theta):
    width_1 = image_size[0]
    print(width_1)

    height_1 = image_size[1]
    print(height_1)
    theta = theta*np.pi/180
    width_2 = width_1 * np.abs(np.cos(theta)) + height_1 * np.abs(np.sin(theta))
    height_2 = width_1 * np.abs(np.sin(theta)) + height_1 * np.abs(np.cos(theta))
    print('gray_rotate_test')
    print(width_2)
    print(height_2)
    center = np.array([0,0])
    center[0] = np.cos(theta) * tar_center[0] - np.sin(theta)*tar_center[1] + 0.5 * width_2 + 0.5*height_1*np.sin(theta)-0.5*width_1*np.cos(theta)
    center[1] = np.sin(theta) * tar_center[0] + np.cos(theta)*tar_center[1] + 0.5 * height_2- 0.5*height_1*np.cos(theta)-0.5*width_1*np.sin(theta)
    print('center')
    print(center)
    return center


def im2c(im,w2c,patch_size):
    
    BB = im[:,:,0]
    GG = im[:,:,1]
    RR = im[:,:,2]
    index_im = 1 + np.floor(RR/8) + 32*np.floor(GG/8) + 32*32*np.floor(BB/8)
    index_im = index_im.astype(np.int32)
    index_im = index_im-1
    out =  np.reshape(w2c[index_im,:],[im.shape[0],im.shape[1],w2c.shape[1]])
    print('out.shape')
    print(w2c[index_im,:].shape)
    np.savetxt('001.txt',out[:,:,2])
    out = transform.resize(out, patch_size,mode='reflect')
    print(out.shape)
    return out

def get_feature_map(im_patch,feature_list,num_feature_ch,out_size,w2c,cell_size,projection_matrix):
    #im_patch_size = im_patch.shape
    #if 'fhog' in feature_list:
    ##    out_size = np.round(np.array([im_patch_size[0],im_patch_size[1]])/cell_size)
    #if 'fhog' not in feature_list:
    #    out_size = np.round(np.array([im_patch_size[0],im_patch_size[1]]))
    out = np.zeros((np.int(out_size[0]),np.int(out_size[1]),num_feature_ch))
    print('out_size')

    print(im_patch.shape)
    channel_id = 0
    if 'gray' in feature_list:
        print('using gray feature')
        im_patch_255 = np.floor(im_patch*255)
        im_patch_255 = im_patch_255.astype(np.uint8)
        dstimg = cv2.cvtColor(im_patch_255,cv2.COLOR_BGR2GRAY)
        
        out[:,:,channel_id] = transform.resize(dstimg, out_size,mode='reflect')
        channel_id = channel_id + 1
    if 'fhog' in feature_list:
        print('using fhog feature')

        hog_feature_t = pyhog.features_pedro(im_patch/255., cell_size)
        img_crop = np.lib.pad(hog_feature_t, ((1, 1), (1, 1), (0, 0)), 'edge')
        img_crop = img_crop[:,:,18:27]
        out[:,:,channel_id:channel_id+9] = img_crop
        channel_id = channel_id + 9
    if 'cn' in feature_list:
        print('using cn feature')
        im_patch_255 = np.floor(im_patch*255)
        cn_out = im2c(im_patch_255,w2c,(np.int(out_size[0]),np.int(out_size[1])))
        cn_out = np.dot(cn_out,projection_matrix)
        out[:,:,channel_id:channel_id+2] = cn_out
    if 'raw' in feature_list:
        print('using raw feature')
        im_patch_255 = np.floor(im_patch*255)
        img_colour = im_patch_255 - im_patch_255.mean()
        out[:,:,channel_id:channel_id+3] = transform.resize(img_colour, out_size,mode='reflect')
    print(out.shape)
    return out



