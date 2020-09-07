#coding:utf-8
from PIL import Image
import os
import sys




#创建底图
target = Image.new('RGBA', (300, 300), (0, 0, 0, 0))
#打开头像
#nike_image = Image.open("./image1.png")
nike_image = Image.open("./0.png")
nike_image = nike_image.resize((300, 300))
#打开装饰
#hnu_image = Image.open("./hnu.png")
hnu_image = Image.open("./water1.png")
hnu_image = hnu_image.resize((300, 300))
# 分离透明通道
#r,g,b,a = hnu_image.split()
r,g,b = hnu_image.split()
# 将头像贴到底图
import pdb
pdb.set_trace()
nike_image.convert("RGBA")
target.paste(nike_image, (0,0))

#将装饰贴到底图
hnu_image.convert("RGBA")
target.paste(hnu_image,(0,0), mask=r*0.1)

# 保存图片
target.save("f.png")
