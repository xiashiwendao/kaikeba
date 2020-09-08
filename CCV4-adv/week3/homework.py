import numpy as np
import os, sys
os.chdir(os.path.dirname(__file__))
import cv2
from matplotlib import pyplot as plt
# 近邻插值法
def nn_insert(data, target_shape):
    src_shape = data.shape
    w = target_shape[0]/src_shape[0]
    newdata = np.zeros(target_shape)
    for i in range(target_shape[1]):
        for j in range(target_shape[0]):
            mapping_i = round(i/w)
            mapping_j = round(j/w)
            if(len(target_shape)==3):
                for n in range(target_shape[2]):
                    newdata[i,j, n] = data[mapping_i, mapping_j, n]
            else:
                newdata[i,j] = data[mapping_i, mapping_j]

            print("i: ", i, "j: ", j, "data: ", "mapping_i: ", mapping_i, "mapping_j: ", mapping_j, newdata[i, j, :])

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
    
