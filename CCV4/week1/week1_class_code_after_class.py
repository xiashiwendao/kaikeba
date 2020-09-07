#!/usr/bin/env python
# coding: utf-8

# In[9]:


a=[1,2,3]


# In[10]:


print("Hello friends，welcome week1's class  .........",29*"-","%s"%(a))


# In[11]:


import cv2
import numpy as np


# # 直接用生成矩阵的方式生成图片

# In[12]:


img0 = np.array([[0,0,1],[0,1,0],[1,0,0]])
#img0 = np.array([[0,0,1],[0,1,0],[1,0,0]])


# # 查看矩阵数值以及大小

# In[13]:


print(img0)


# In[17]:


print(img0.shape)
print("img0 size = %s,%s"%(img0.shape[0],img0.shape[1]))


# In[7]:


import matplotlib.pyplot as plt
# import matplotlib.pylab as plt
plt.imshow(img0,cmap = 'gray' )
# color map 
#plt.imshow(img0)

# pycharm要加一句：plt.show() 
# 加一个%matplotlib inline就会显示


# # 彩色图像的颜色空间转换

# In[ ]:


# https://blog.csdn.net/zhang_cherry/article/details/88951259


# # 从摄像头采集图像

# In[98]:


# read camera
#cap = cv2.VideoCapture(0)
# read video
#cap  = cv2.VideoCapture("/Users/zhaomingming/Documents/HTC/核心课/CVFundamentals/week1/How Computer Vision Works.mp4")
#cap = cv2.VideoCapture("../How Computer Vision Works.mp4")
cap = cv2.VideoCapture("How Computer Vision Works.mp4")



# In[99]:


print(cap.isOpened())


# In[ ]:


return_value=True
while return_value:
    return_value,frame = cap.read()
    print(cap.isOpened())

    # frame 图像帧
    #cv2.cvtColor()
    #cv2.COLOR_BGR2RGB
    plt.imshow(frame,cmap = 'gray')
 
    #plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# # frame 的大小是多少？

# In[66]:


print(frame.shape)


# In[75]:


# 关闭capture
cap.release()


# In[ ]:





# # 从文件读取图像数据

# In[8]:


img  = cv2.imread("lena.jpg")

#cv2.cvtColor()
#cv2.COLOR_BGR2RGB
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# In[86]:


print(img.shape)


# In[87]:


# range of instrest


# In[88]:


roi = img[100:200,300:400]


# In[89]:


plt.imshow(img)
#plt.imshow(roi)


# In[230]:


# 黑白图像


# In[96]:


img_gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_gray.shape


# In[95]:


plt.imshow(img_gray,cmap='gray')


# In[ ]:


#  彩色图像


# In[ ]:





# # rgb2hsv

# In[10]:


import cv2
img_BGR = cv2.imread('lena.jpg')
img_hsv=cv2.cvtColor(img_BGR,cv2.COLOR_BGR2HSV)
plt.imshow(img_hsv,cmap='')


# In[ ]:


# 肤色检测


# In[5]:


# 二值化
import cv2
img =  cv2.imread('lena.jpg')


# In[6]:


plt.imshow(cv2.threshold(img,128,200,cv2.THRESH_BINARY))


# In[228]:


# 图像的放大与缩小


# In[15]:


# 列，行
img =cv2.resize(img,(50,30))
plt.imshow(img)
img.shape


# In[18]:


# 列，行
img =  cv2.imread('lena.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img =cv2.resize(img,(500,300))
plt.imshow(img)
img.shape


# In[22]:


# roi
img_roi =img[100:300,0:200]
plt.imshow(img_roi)


# In[31]:


import numpy as np
# 移动：
M = np.float32([[1,0,300],[0,1,200]])

print(img[0+50,0])
img_1=cv2.warpAffine(img,M,(1000,1000))

plt.imshow(img_1)
print(img_1[200+50,300])


# In[229]:


# 图像的旋转变换与拉伸变换


# In[56]:


#pts1 = np.float32([[50,50],[200,50],[50,200]])
#pts2 = np.float32([[10,100],[200,50],[100,250]])
#M = cv2.getAffineTransform(pts1,pts2)
#print(M)
#180/3.14
theta=0.5
#M = np.float32([[np.cos(theta),-np.sin(theta),100],[np.sin(theta),np.cos(theta),200]])
M = np.float32([[0.1,0,100],[0,2,100]])
# 变换矩阵，平移，斜切，旋转
# affine
cols=800
rows=800
dst = cv2.warpAffine(img,M,(cols,rows))
plt.imshow(dst)


# In[59]:


pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[100,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
print(M)
# 拉伸变换后者透视变换
dst = cv2.warpPerspective(img,M,(300,300))
plt.imshow(dst)


# In[261]:


# 图像模糊与锐化


# In[62]:


img= cv2.GaussianBlur(img, (11, 11), 1, 0)  # 高斯模糊
plt.imshow(img)


# In[ ]:


cv2.Canny(pil_img3,30,150)


# # 图像滤波/卷积

# In[139]:


kernel = np.ones((3,3),np.float32)/8
kernel=-kernel
kernel[0,:]=[-1,-1,-1]
kernel[1,:]=[0,0,0]
kernel[2,:]=[1,1,1]


print(kernel)
plt.imshow(img)


# In[140]:


#dst=cv.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]])；当ddepth=-1时，表示输出图像与原图像有相同的深度。
print(img.shape)
result = cv2.filter2D(img,-1,kernel)
result.shape
print(result[0,0])
plt.imshow(result*255)


# In[118]:


result = cv2.filter2D(result,-1,kernel)

plt.imshow(result)
result.shape


# In[101]:


# 形态学运算


# In[ ]:





# In[ ]:





# In[ ]:


# 灰度直方图


# In[ ]:





# In[ ]:





# In[ ]:


# 直方图均衡化


# In[ ]:





# In[ ]:





# In[ ]:


# 加水印，保护版权


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[3]:


import torch


# In[10]:


import matplotlib


# In[11]:


get_ipython().system(u'pip install matplotlib -i https://pypi.tuna.tsinghua.edu.cn/simple')


# In[13]:


from image_process import *
image_data = generate_data()


# # matplotlib 使用例子
# https://www.cnblogs.com/yanghailin/p/11611333.html,

# In[14]:


import matplotlib.pyplot as plt


# In[ ]:


i=0


# In[120]:


print(image_data[i%10])
plt.imshow(image_data[i%10],cmap = 'gray')
i=i+1


# In[83]:


plt.imshow(cv2.imread('lena.jpg'))


# In[148]:


img  = cv2.imread("lena.jpg")
img= cv2.cvtColor(img, cv2.COLOR_BGR2RGB)



# # add watermask

# In[152]:


wm = cv2.imread("water1.png")
wm = cv2.resize(wm,(300,300))
wm = 255-wm
img1 = cv2.resize(img,(300,300))
#img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
print(wm.shape)
plt.imshow(cv2.add(wm,img1))

plt.imshow(cv2.addWeighted(wm,0.9,img1,0.5,0))


#plt.imshow(wm)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




