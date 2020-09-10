import numpy as np
import os, sys
os.chdir(os.path.dirname(__file__))
import cv2
from matplotlib import pyplot as plt
# 近邻插值法
def nn_insert(data, target_shape):
    srcH, srcW, _ = data.shape
    dstH, dstW = target_shape[0], target_shape[1]
    w_h = dstH/srcH #target_shape[0]/src_shape[0]
    w_w = dstW/srcW #target_shape[1]/src_shape[1]
    # 这里注意，像素矩阵必须要指定dtype=np.unit，否则图像不能正常显示
    # 因为像素必须要是整数
    newdata = np.zeros((dstH, dstW, 3), dtype=np.uint8)
    for i in range(target_shape[0]-1):
        for j in range(target_shape[1]-1):
            mapping_i = round(i*srcH/dstH) #round(i/w_h)
            mapping_j = round(j*srcH/dstH) #round(j/w_w)
            newdata[i,j] = data[mapping_i, mapping_j]

    return newdata

# 双线性插值法
def bilinear_interpolate_slow(src, dst_size):
    height_src, width_src, channel_src = src.shape  # (h, w, ch)
    height_dst, width_dst = dst_size  # (h, w)
    
    """
    中心对齐，投影目标图的横轴和纵轴到原图上
    """
    ws_p = np.array([(i + 0.5) / width_dst * width_src - 0.5 for i in range(width_dst)], dtype=np.float32)
    hs_p = np.array([(i + 0.5) / height_dst * height_src - 0.5 for i in range(height_dst)], dtype=np.float32)
    ws_p = np.clip(ws_p, 0, width_src-1)  # 实验发现要这样子来一下才能跟torch的输出结果一致
    hs_p = np.clip(hs_p, 0, height_src-1)
    
    """找出每个投影点在原图横轴方向的近邻点坐标对"""
    # w_0的取值范围是 0 ~ (width_src-2)，因为w_1 = w_0 + 1
    ws_0 = np.clip(np.floor(ws_p), 0, width_src-2).astype(np.int)
        
    """找出每个投影点在原图纵轴方向的近邻点坐标对"""
    # h_0的取值范围是 0 ~ (height_src-2)，因为h_1 = h_0 + 1
    hs_0 = np.clip(np.floor(hs_p), 0, height_src-2).astype(np.int)
        
    """
    计算目标图各个点的像素值
    f(h, w) = f(h_0, w_0) * (1 - u) * (1 - v)
            + f(h_0, w_1) * (1 - u) * v
            + f(h_1, w_0) * u * (1 - v)
            + f(h_1, w_1) * u * v
    """
    dst = np.zeros(shape=(height_dst, width_dst, channel_src), dtype=np.int8)
    us = hs_p - hs_0
    vs = ws_p - ws_0
    _1_us = 1 - us
    _1_vs = 1 - vs
    for h in range(height_dst):
        h_0, h_1 = hs_0[h], hs_0[h]+1  # 原图的坐标
        for w in range(width_dst):
            w_0, w_1 = ws_0[w], ws_0[w]+1 # 原图的坐标
            for c in range(channel_src):
                dst[h][w][c] = src[h_0][w_0][c] * _1_us[h] * _1_vs[w] \
                            + src[h_0][w_1][c] * _1_us[h] * vs[w] \
                            + src[h_1][w_0][c] * us[h] * _1_vs[w] \
                            + src[h_1][w_1][c] * us[h] * vs[w]
    return dst


import numpy as np
import math

# 双线性插值法(通过numpy进行加速，效果失败，出现很多重影现象)
def bilinear_interpolate(src, dst_size):
    height_src, width_src, channel_src = src.shape  # (h, w, ch)
    height_dst, width_dst = dst_size  # (h, w)

    """中心对齐，投影目标图的横轴和纵轴到原图上"""
    ws_p = np.array([(i + 0.5) / width_dst * width_src - 0.5 for i in range(width_dst)], dtype=np.float32)
    hs_p = np.array([(i + 0.5) / height_dst * height_src - 0.5 for i in range(height_dst)], dtype=np.float32)
    ws_p = np.clip(ws_p, 0, width_src-1)  # 实验发现要这样子来一下才能跟torch的输出结果一致
    hs_p = np.clip(hs_p, 0, height_src-1)
    ws_p = np.repeat(ws_p.reshape(1, width_dst), height_dst, axis=0)
    hs_p = np.repeat(hs_p.reshape(height_dst, 1), width_dst, axis=1)

    """找出每个投影点在原图的近邻点坐标"""
    ws_0 = np.clip(np.floor(ws_p), 0, width_src - 2).astype(np.int)
    hs_0 = np.clip(np.floor(hs_p), 0, height_src - 2).astype(np.int)
    ws_1 = ws_0 + 1
    hs_1 = hs_0 + 1

    """四个临近点的像素值"""
    f_00 = src[hs_0, ws_0, :].T
    f_01 = src[hs_0, ws_1, :].T
    f_10 = src[hs_1, ws_0, :].T
    f_11 = src[hs_1, ws_1, :].T

    """计算权重"""
    w_00 = ((hs_1 - hs_p) * (ws_1 - ws_p)).T
    w_01 = ((hs_1 - hs_p) * (ws_p - ws_0)).T
    w_10 = ((hs_p - hs_0) * (ws_1 - ws_p)).T
    w_11 = ((hs_p - hs_0) * (ws_p - ws_0)).T

    """计算目标像素值"""
    return (f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 * w_11).T
    # return np.array([(f_00 * w_00).T + (f_01 * w_01).T + (f_10 * w_10).T + (f_11 * w_11).T], dtype=np.int8)

# 双线性简洁实现版本（效率并不高）
def bilinear_interpolate(img,target_shape):
    dstH, dstW = target_shape
    scrH,scrW,_=img.shape
    img=np.pad(img,((0,1),(0,1),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=(i+1)*(scrH/dstH)-1
            scry=(j+1)*(scrW/dstW)-1
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            retimg[i,j]=(1-u)*(1-v)*img[x,y]+u*(1-v)*img[x+1,y]+(1-u)*v*img[x,y+1]+u*v*img[x+1,y+1]
    return retimg

def BiBubic(x):
    x=abs(x)
    if x<=1:
        return 1-2*(x**2)+(x**3)
    elif x<2:
        return 4-8*x+5*(x**2)-(x**3)
    else:
        return 0

# 双三次插值
def BiCubic_interpolation(img,target_shape):
    scrH,scrW,_=img.shape
    dstH, dstW = target_shape

    #img=np.pad(img,((1,3),(1,3),(0,0)),'constant')
    retimg=np.zeros((dstH,dstW,3),dtype=np.uint8)
    for i in range(dstH):
        for j in range(dstW):
            scrx=i*(scrH/dstH)
            scry=j*(scrW/dstW)
            x=math.floor(scrx)
            y=math.floor(scry)
            u=scrx-x
            v=scry-y
            tmp=0
            for ii in range(-1,2):
                for jj in range(-1,2):
                    if x+ii<0 or y+jj<0 or x+ii>=scrH or y+jj>=scrW:
                        continue
                    tmp+=img[x+ii,y+jj]*BiBubic(ii-u)*BiBubic(jj-v)
            retimg[i,j]=np.clip(tmp,0,255)
    return retimg

def main_nn():
    image = plt.imread('undist6.jpg')
    image_nn = nn_insert(image, (900, 1400, 3))
    plt.subplot(231)
    plt.title("my nn image")
    plt.imshow(image_nn)

    plt.subplot(232)
    plt.title("raw image")
    plt.imshow(image)

    image_cv2 = cv2.resize(image, (1400, 900), interpolation=cv2.INTER_NEAREST)
    plt.subplot(233)
    plt.title("cv2 resize image")
    plt.imshow(image_cv2)

    image_twoline = bilinear_interpolate(image, (900, 1400))
    plt.subplot(234)
    plt.title("two line insert")
    plt.imshow(image_twoline)
    
    image_twiceThreeline = BiCubic_interpolation(image, (900, 1400))
    plt.subplot(235)
    plt.title("twice three time line insert")
    plt.imshow(image_twoline)
    plt.show()

if __name__ == "__main__":
    main_nn()

    # 双线性插值    
    # main_twoline_insert()