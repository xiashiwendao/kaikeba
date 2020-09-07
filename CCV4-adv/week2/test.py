import numpy as np
import tensorflow as tf
import os

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto() #对session进行参数配置
config.allow_soft_placement=False #如果你指定的设备不存在，允许TF自动分配设备
config.gpu_options.per_process_gpu_memory_fraction=1 #分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = False #按需分配显存，这个比较重要


# m = np.random.rand(1,3,1,3)
n = np.random.rand(1, 2, 1, 3)
# print(m)
# m = np.array([[[[2,6,7]],[[5,1,9]],[[2,-1,-5]]]],dtype=np.float)
m = np.array([[[[2,5,2]],[[6,1,-1]],[[7,9,-5]]]],dtype=np.float)
print("m.shape: ", m.shape)
n = np.array([[[[-1,5,4]],[[2,1,6]]]], dtype=np.float)
print("n.shape: ", n.shape)
# n = np.array([[[[-1,5,4],[2,1,6]]]], dtype=np.float)
input_data = tf.Variable(m, dtype=np.float)
filter_data = tf.Variable(n, dtype=np.float)
# p = tf.nn.conv2d(m.T, n, strides=[1,1,1,1], padding="SAME")
p = tf.nn.conv2d(input_data, filter_data, strides=[1,1], padding="SAME")
print("conv result: ", p)

with tf.Session(config=config) as sess:
    tf.global_variables_initializer().run(session=sess)
    sess.run(p)
    print("conv result: ", p.eval())