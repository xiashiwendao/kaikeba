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


# In[227]:


import matplotlib.pyplot as plt
plt.imshow(img0,cmap = 'gray' )


# # 彩色图像的颜色空间转换

# In[ ]:


# https://blog.csdn.net/zhang_cherry/article/details/88951259


# # 从摄像头采集图像

# In[189]:


#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("../How Computer Vision Works.mp4")


# In[226]:


ret,frame = cap.read()
print(cap.isOpened())
plt.imshow(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))


# In[173]:


cap.release()


# In[ ]:





# # 从文件读取图像数据

# In[139]:


img  = cv2.imread("lena.jpg")


# In[144]:


print(img.shape)


# In[ ]:


# range of instrest


# In[ ]:


roi = img[100:200,300:400]


# In[230]:


# 黑白图像


# In[ ]:


#  彩色图像


# In[231]:


img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB


# In[ ]:


cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)


# In[ ]:


cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV


# In[ ]:


# 肤色检测


# In[235]:


# 二值化


# In[242]:


plt.imshow(cv2.threshold(img,128,200,cv2.THRESH_BINARY))


# In[228]:


# 图像的放大与缩小


# In[243]:


plt.imshow(cv2.resize(img,(300,500)))


# In[247]:


# 移动：
M = np.float32([[1,0,30],[0,1,30]])

plt.imshow(cv2.warpAffine(img,M,(500,300)))


# In[229]:


# 图像的旋转与拉伸


# In[260]:


pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv2.getAffineTransform(pts1,pts2)
print(M)
theta=1
M = np.float32([[np.cos(theta),-np.sin(theta),500],[np.sin(theta),np.cos(theta),100]])
cols=800
rows=800
dst = cv2.warpAffine(img,M,(cols,rows))
plt.imshow(dst)


# In[ ]:


pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img,M,(300,300))


# In[261]:


# 图像模糊与锐化


# In[ ]:


lena_gaussian_blur = cv2.GaussianBlur(lena_RGB, (5, 5), 1, 0)  # 高斯模糊


# In[ ]:


cv2.Canny(pil_img3,30,150)


# In[ ]:


kernel = np.ones((9,9),np.float32)/81
result = cv2.filter2D(img,-1,kernel)


# In[ ]:


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


# In[129]:


img  = cv2.imread("lena.jpg")




# # add watermask

# In[234]:


wm = cv2.imread("water1.png")
wm = cv2.resize(wm,(300,300))
wm = 255-wm
img1 = cv2.resize(img,(300,300))
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
print(wm.shape)
plt.imshow(cv2.add(wm,img1))

plt.imshow(cv2.addWeighted(wm,1,img1,0.2,0))


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




