# - * - coding: utf - 8 -*-
import tensorflow as tf
import os
import numpy as np
 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
 
# tf.nn.convolution
# 计算N维卷积的和
 
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)
y = tf.nn.convolution(input_data, filter_data, strides=[1, 1], padding='SAME')
 
print('1. tf.nn.convolution : ', y)
# 1. tf.nn.convolution :  Tensor("convolution:0", shape=(10, 9, 9, 2), dtype=float32)
 
 
# tf.nn.conv2d
# 对一个思维的输入数据 input 和四维的卷积核filter 进行操作,然后对输入的数据进行二维的卷积操作,得到卷积之后的结果
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)
y = tf.nn.conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
 
print('2. tf.nn.conv2d : ', y)
#2. tf.nn.conv2d :  Tensor("Conv2D:0", shape=(10, 9, 9, 2), dtype=float32)
 
# tf.nn.depthwise_conv2d
# input 的数据维度 [batch ,in_height,in_wight,in_channels]
# 卷积核的维度是 [filter_height,filter_heught,in_channel,channel_multiplierl]
# 讲不通的卷积和独立的应用在in_channels 的每一个通道上(从通道 1 到通道channel_multiplier)
# 然后将所有结果进行汇总,输出通道的总数是,in_channel * channel_multiplier
 
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
filter_data = tf.Variable(np.random.rand(2, 2, 3, 2), dtype=np.float32)
 
y = tf.nn.depthwise_conv2d(input_data, filter_data, strides=[1, 1, 1, 1], padding='SAME')
print('3. tf.nn.depthwise_conv2d : ', y)
 
# tf.nn.separable_conv2d
# 利用几个分离的卷积核去做卷积,在该函数中,将应用一个二维的卷积核,在每个通道上,以深度channel_multiplier进行卷积
input_data = tf.Variable(np.random.rand(10, 9, 9, 3), dtype=np.float32)
depthwise_filter = tf.Variable(np.random.rand(2, 2, 3, 5), dtype=np.float32)
poinwise_filter = tf.Variable(np.random.rand(1, 1, 15, 20), dtype=np.float32)
# out_channels >= channel_multiplier * in_channels
y = tf.nn.separable_conv2d(input_data, depthwise_filter=depthwise_filter, pointwise_filter=poinwise_filter,
                           strides=[1, 1, 1, 1], padding='SAME')
print('4. tf.nn.separable_conv2d : ', y)
 
# 计算Atrous卷积,又称孔卷积或者扩张卷积
input_data = tf.Variable(np.random.rand(1, 5, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(3, 3, 1, 1), dtype=np.float32)
y = tf.nn.atrous_conv2d(input_data, filters, 2, padding='SAME')
print('5. tf.nn.atrous_conv2d : ', y)
 
# 在解卷积网络(deconvolutional network) 中有时被称为'反卷积',但实际上是conv2d的转置,而不是实际的反卷积
x = tf.random_normal(shape=[1, 3, 3, 1])
kernal = tf.random_normal(shape=[2, 2, 3, 1])
y = tf.nn.conv2d_transpose(x, kernal, output_shape=[1, 5, 5, 3], strides=[1, 2, 2, 1], padding='SAME')
print('6. tf.nn.conv2d_transpose : ', y)
 
# 与二维卷积类似,用来计算给定三维输入和过滤器的情况下的一维卷积.
# 不同的是,它的输入维度为 3,[batch,in_width,in_channels].
# 卷积核的维度也是三维,[filter_height,in_channel,channel_multiplierl]
# stride 是一个正整数,代表一定每一步的步长
input_data = tf.Variable(np.random.rand(1, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(3, 1, 3), dtype=np.float32)
y = tf.nn.conv1d(input_data, filters, stride=2, padding='SAME')
print('7. tf.nn.conv1d : ', y)
 
# 与二维卷积类似,用来计算给定五维输入和过滤器的情况下的三维卷积.
# 不同的是,它的输入维度为 5,[batch,in_depth,in_height,in_width,in_channels].
# 卷积核的维度也是三维,[filter_depth,filter_height,in_channel,channel_multiplierl]
# stride 相较二维卷积多了一维,变为[strides_batch,strides_depth,strides_height,strides_width,strides_channel],必须保证strides[0] = strides[4] =1
input_data = tf.Variable(np.random.rand(1, 2, 5, 5, 1), dtype=np.float32)
filters = tf.Variable(np.random.rand(2, 3, 3, 1, 3), dtype=np.float32)
y = tf.nn.conv3d(input_data, filters, strides=[1, 2, 2, 1, 1], padding='SAME')
print('8. tf.nn.conv3d : ', y)
 
# 与conv2d_transpose 二维反卷积类似
# 在解卷积网络(deconvolutional network) 中有时被称为'反卷积',但实际上是conv3d的转置,而不是实际的反卷积
x = tf.random_normal(shape=[2, 1, 3, 3, 1])
kernal = tf.random_normal(shape=[2, 2, 2, 3, 1])
y = tf.nn.conv3d_transpose(x, kernal, output_shape=[2, 1, 5, 5, 3], strides=[1, 2, 2, 2, 1], padding='SAME')
print('9. tf.nn.conv3d_transpose : ', y)