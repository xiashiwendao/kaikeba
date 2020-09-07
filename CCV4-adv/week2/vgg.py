# 参考：https://blog.csdn.net/zhangwei15hh/article/details/78417789

from  datetime import datetime
import tensorflow as tf
import math
import time
import os, sys

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0" #选择哪一块gpu,如果是-1，就是调用cpu
config = tf.ConfigProto() #对session进行参数配置
config.allow_soft_placement=True #如果你指定的设备不存在，允许TF自动分配设备
# config.gpu_options.per_process_gpu_memory_fraction=0.7 #分配百分之七十的显存给程序使用，避免内存溢出，可以自己调整
config.gpu_options.allow_growth = True #按需分配显存，这个比较重要

batch_size = 2
num_batches = 100

# 用来创建卷积层并把本层的参数存入参数列表
# input_op:输入的tensor 
# name:该层的名称 
# kh:卷积层的高，kw:卷积层的宽 
# n_out:输出通道数
# dh:步长的高 dw:步长的宽
# p是参数列表
def conv_op(input_op,name,kh,kw,n_out,dh,dw,p):
    # 输入的通道数
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope + "w",shape=[kh,kw,n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        conv = tf.nn.conv2d(input_op, kernel, (1,dh,dw,1),padding='SAME')
        bias_init_val = tf.constant(0.0, shape=[n_out],dtype=tf.float32)
        biases = tf.Variable(bias_init_val , trainable=True , name='b')
        z = tf.nn.bias_add(conv,biases)
        activation = tf.nn.relu(z,name=scope)
        p += [kernel,biases]
        return activation

# 定义全连接层
def fc_op(input_op,name,n_out,p):
    n_in = input_op.get_shape()[-1].value
    with tf.name_scope(name) as scope:
        kernel = tf.get_variable(scope+'w',shape=[n_in,n_out],dtype=tf.float32,initializer=tf.contrib.layers.xavier_initializer_conv2d())
        biases = tf.Variable(tf.constant(0.1,shape=[n_out],dtype=tf.float32),name='b')
        # tf.nn.relu_layer()用来对输入变量input_op与kernel做乘法并且加上偏置b
        activation = tf.nn.relu_layer(input_op,kernel,biases,name=scope)
        p += [kernel,biases]
        return activation

# 定义最大池化层
def mpool_op(input_op,name,kh,kw,dh,dw):
    return tf.nn.max_pool(input_op,ksize=[1,kh,kw,1],strides=[1,dh,dw,1],padding='SAME',name=name)

#定义网络结构
def inference_op(input_op,keep_prob):
    p = []
    conv1_1 = conv_op(input_op,name='conv1_1',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    conv1_2 = conv_op(conv1_1,name='conv1_2',kh=3,kw=3,n_out=64,dh=1,dw=1,p=p)
    pool1 = mpool_op(conv1_2,name='pool1',kh=2,kw=2,dw=2,dh=2)

    conv2_1 = conv_op(pool1,name='conv2_1',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    conv2_2 = conv_op(conv2_1,name='conv2_2',kh=3,kw=3,n_out=128,dh=1,dw=1,p=p)
    pool2 = mpool_op(conv2_2, name='pool2', kh=2, kw=2, dw=2, dh=2)

    conv3_1 = conv_op(pool2, name='conv3_1', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_2 = conv_op(conv3_1, name='conv3_2', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    conv3_3 = conv_op(conv3_2, name='conv3_3', kh=3, kw=3, n_out=256, dh=1, dw=1, p=p)
    pool3 = mpool_op(conv3_3, name='pool3', kh=2, kw=2, dw=2, dh=2)

    conv4_1 = conv_op(pool3, name='conv4_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_2 = conv_op(conv4_1, name='conv4_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv4_3 = conv_op(conv4_2, name='conv4_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool4 = mpool_op(conv4_3, name='pool4', kh=2, kw=2, dw=2, dh=2)

    conv5_1 = conv_op(pool4, name='conv5_1', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_2 = conv_op(conv5_1, name='conv5_2', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    conv5_3 = conv_op(conv5_2, name='conv5_3', kh=3, kw=3, n_out=512, dh=1, dw=1, p=p)
    pool5 = mpool_op(conv5_3, name='pool5', kh=2, kw=2, dw=2, dh=2)

    shp = pool5.get_shape()
    flattened_shape = shp[1].value * shp[2].value * shp[3].value
    resh1 = tf.reshape(pool5,[-1,flattened_shape],name="resh1")

    fc6 = fc_op(resh1,name="fc6",n_out=4096,p=p)
    fc6_drop = tf.nn.dropout(fc6,keep_prob,name='fc6_drop')
    fc7 = fc_op(fc6_drop,name="fc7",n_out=4096,p=p)
    fc7_drop = tf.nn.dropout(fc7,keep_prob,name="fc7_drop")
    fc8 = fc_op(fc7_drop,name="fc8",n_out=1000,p=p)
    softmax = tf.nn.softmax(fc8)
    predictions = tf.argmax(softmax,1)
    return predictions,softmax,fc8,p

def time_tensorflow_run(session,target,feed,info_string):
    num_steps_burn_in = 10  # 预热轮数
    total_duration = 0.0  # 总时间
    total_duration_squared = 0.0  # 总时间的平方和用以计算方差
    for i in range(num_batches + num_steps_burn_in):
        start_time = time.time()
        _ = session.run(target,feed_dict=feed)
        duration = time.time() - start_time
        if i >= num_steps_burn_in:  # 只考虑预热轮数之后的时间
            if not i % 10:
                print('%s:step %d,duration = %.3f' % (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration
                total_duration_squared += duration * duration
    mn = total_duration / num_batches  # 平均每个batch的时间
    vr = total_duration_squared / num_batches - mn * mn  # 方差
    sd = math.sqrt(vr)  # 标准差
    print('%s: %s across %d steps, %.3f +/- %.3f sec/batch' % (datetime.now(), info_string, num_batches, mn, sd))

def run_benchmark():
    with tf.Graph().as_default():
        image_size = 224  # 输入图像尺寸
        images = tf.Variable(tf.random_normal([batch_size, image_size, image_size, 3], dtype=tf.float32, stddev=1e-1))
        keep_prob = tf.placeholder(tf.float32)
        prediction,softmax,fc8,p = inference_op(images,keep_prob)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=config)
        sess.run(init)
        time_tensorflow_run(sess, prediction,{keep_prob:1.0}, "Forward")
        # 用以模拟训练的过程
        objective = tf.nn.l2_loss(fc8)  # 给一个loss
        grad = tf.gradients(objective, p)  # 相对于loss的 所有模型参数的梯度
        time_tensorflow_run(sess, grad, {keep_prob:0.5},"Forward-backward")


run_benchmark()