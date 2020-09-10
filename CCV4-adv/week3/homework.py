import numpy as np
import os, sys
os.chdir(os.path.dirname(__file__))
import cv2
from matplotlib import pyplot as plt
# 近邻插值法
def nn_insert_raw(data, target_shape):
    src_shape = data.shape
    w_h = target_shape[0]/src_shape[0]
    w_w = target_shape[1]/src_shape[1]
    newdata = np.zeros(target_shape)
    for i in range(target_shape[0]-1):
        for j in range(target_shape[1]-1):
            mapping_i = round(i/w_h)
            mapping_j = round(j/w_w)
            newdata[i,j] = data[mapping_i, mapping_j]

    return newdata

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

if __name__ == "__main__":
    p = (1,2)
    print(p[0])
    image = plt.imread('undist6.jpg')
    # plt.imshow(image)
    # plt.show()
    # image = np.array([[1,2],[3,4]])
    image_nn = nn_insert(image, (900, 1400, 3))
    # image_nn = nn_insert(image, (4,4))
    # print(image_nn)
    plt.subplot(131)
    # plt.figure(figsize=(15, 15))
    plt.title("my nn image")
    plt.imshow(image_nn)

    plt.subplot(132)
    # plt.figure(figsize=(15, 15))
    plt.title("raw image")
    plt.imshow(image)

    image_cv2 = cv2.resize(image, (1400, 900), interpolation=cv2.INTER_NEAREST)
    plt.subplot(133)
    plt.title("cv2 resize image")
    plt.imshow(image_cv2)
    plt.show()
    