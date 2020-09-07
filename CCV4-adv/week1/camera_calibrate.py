#coding:utf-8
"""
相机校正
学会用棋盘格进行相机校正（张正友标定法）
熟悉opencv相关函数
cv2.findChessboardCorners
cv2.cornerSubPix
cv2.drawChessboardCorners
cv2.calibrateCamera
cv2.undistort
"""
import numpy as np
import cv2
import os

def calibrate_camera():
    #每个校准图像映射到棋盘角到数量
    objPoints = {
        1: (9, 6),
        2: (9, 6),
        3: (9, 6),
        4: (9, 6),
        5: (9, 6),
        6: (9, 6),
        7: (9, 6),
        8: (9, 6),
        9: (9, 6),
        10: (9, 6),
        11: (9, 6),
        12: (9, 6),
        13: (9, 6),
        14: (9, 6),
        15: (9, 6),
        16: (9, 6),
        17: (9, 6),
        18: (9, 6),
        19: (9, 6),
        20: (9, 6),
    }
    #目标点空间坐标
    obj3DList = []
    
    #图像中棋盘格中的2D点
    obj2DList = []

    for k in objPoints:
        nx, ny = objPoints[k]
        ######棋盘格对应3D坐标点，x为0-8， y为0-4（对应棋盘格横着9个点，纵着5个点）， z = 0
        obj = np.zeros((nx * ny, 3), np.float32)

        obj[:, :2] = np.mgrid[0: nx, 0: ny].T.reshape(-1, 2)
        img_folder = r"C:\Users\Lorry\MySpace\code_Space\learn_kaikeba\CCV4-adv\week1HomeWork\camera_cal_pic"
        fname = os.path.join(img_folder,'calibration%s.jpg' % str(k))
        if os.path.exists(fname):
            print("file: ", fname, " exists!")
        else:
            print("file: ", fname, " don't exists")
            continue
        img = cv2.imread(fname)
        #将图像转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        ########查找角点，利用cv2.findChessboardCorners函数，函数返回
        ####ret：是否查找到； corners：角点坐标
        #####################填空1 （一行代码）#####################################
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny))
        
        #########################################################################

        if ret == True:
            print("INFO: get all the corners")
            obj3DList.append(obj)
            #利用cv2.cornerSubPix可以更精细的查找角点坐标，如果查找到了，用这个，没查找到用cv2.findChessboardCorners中找到的角点
            corners2 = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria=(cv2.TERM_CRITERIA_MAX_ITER | cv2.TERM_CRITERIA_EPS, 30, 0.001))
            if corners2.any():
                obj2DList.append(corners2)
            else:
                obj2DList.append(corners)
            #可以利用cv2.drawChessboardCorners显示每张图查找到的角点的坐标
            # cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
            # cv2.imshow("img", img)
            # cv2.waitKey(0)
        else:
            print('Warning: ret = %s for %s' % (ret, fname))
    test_img_folder = r'C:\Users\Lorry\MySpace\code_Space\learn_kaikeba\CCV4-adv\week1HomeWork\testImage'
    img = cv2.imread(os.path.join(test_img_folder, 'straight_lines1.jpg'))
    img_size = (img.shape[1], img.shape[0])
    #利用图像中2d点和空间3d点计算旋转和平移矩阵，函数使用cv2.calibrateCamera，返回mtx（相机内参矩阵）, dist（畸变矩阵）
    ################填空2（一行代码）################################################
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj3DList, obj2DList, gray.shape[::-1], None, None)
    ##############################################################################
    

    return mtx, dist

mtx, dist = calibrate_camera()
print ("mtx, dist", mtx, dist)
img = cv2.imread('./camera_cal_pic/calibration1.jpg')

##########将img进行校正，利用cv2.undistort这个函数，根据相机内参和外参进行相机校正，校正后的图像为dst

####################填空3（一行代码）#################################################
dst = cv2.undistort(img, mtx, dist, None)
###################################################################################

cv2.imwrite('./camera_cal_pic_undistort/calibration1_undistort.jpg', dst)
