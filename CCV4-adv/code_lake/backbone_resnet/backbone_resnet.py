import tensorflow as tf
slim = tf.contrib.slim

ResNet_demo = {"layer_50": [{"depth": 256, "num_class": 3}, {"depth": 512, "num_class": 4}, {"depth": 1024, "num_class": 6}, {"depth": 2048, "num_class": 3}],
               "layer_101": [{"depth": 256, "num_class": 3}, {"depth": 512, "num_class": 4}, {"depth": 1024, "num_class": 23}, {"depth": 2048, "num_class": 3}],
               "layer_152": [{"depth": 256, "num_class": 3}, {"depth": 512, "num_class": 8}, {"depth": 1024, "num_class": 36}, {"depth": 2048, "num_class": 3}]}


def sampling(input_tensor,ksize=1,stride=2): 
    '''
    input_tensor,  # Tensor入口
    ksize=1,  # 采样块大小
    stride=2):  # 采样步长
    '''
    data = input_tensor
    data = slim.max_pool2d(data, ksize, stride=stride)
    return data


def depthFilling(input_tensor, depth): 
    '''
    input_tensor,  # 输入Tensor                        
    depth):  # 输出深度
    '''
    data = input_tensor  # 取出输入tensor的深度
    input_depth = data.get_shape().as_list()[3]
    # tf.pad用与维度填充，不理解的同学可以去TensoFLow官网了解一下
    data = tf.pad(data, [[0, 0],[0, 0],[0, 0],[abs(depth - input_depth)//2, abs(depth - input_depth)//2]])
    return data


def bottleneck(input_tensor, output_depth):
    # 取出通道
    redepth = input_tensor.get_shape().as_list()[3]
    # 当通道不相符时，进行全零填充并降采样
    if output_depth != redepth:
       # 全零填充
        input_tensor = depthFilling(input_tensor, output_depth)
        # 降采样
        input_tensor = sampling(input_tensor)
    data = input_tensor
    # 降通道处理
    data = slim.conv2d(inputs=data, num_outputs=output_depth//4, kernel_size=1, stride=1)
    # 提取特征
    data = slim.conv2d(inputs=data, num_outputs=output_depth//4, kernel_size=3, stride=1)
    # 通道还原
    data = slim.conv2d(inputs=data, num_outputs=output_depth, kernel_size=1, stride=1, activation_fn=None, normalizer_fn=None)
    # 生成残差
    data = data + input_tensor
    data = tf.nn.relu(data)
    return data


def cnn_to_fc(input_tensor, num_output, train=False, regularizer=None):
    '''
    input_tensor,  # Tensor入口
    num_output,  # 输出接口数量
    train=False,  # 是否使用dropout
    regularizer=None):  # 正则函数
    '''
    data = input_tensor  # 得到输出信息的维度，用于全连接层的输入
    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    reshaped = tf.reshape(data, [data_shape[0], nodes])
    # 最后全连接层
    with tf.variable_scope('layer-fc'):
        fc_weights = tf.get_variable("weight",[nodes, num_output], initializer=tf.truncated_normal_initializer(stddev=0.1))
        if regularizer != None:
            tf.add_to_collection('losses', regularizer(fc_weights))
        fc_biases = tf.get_variable("bias", [num_output], initializer=tf.constant_initializer(0.1))
    fc = tf.nn.relu(tf.matmul(reshaped, fc_weights) + fc_biases)
    if train:
        fc = tf.nn.dropout(fc, 0.5)
    return fc

# 堆叠ResNet模块
def inference(input_tensor, demos, num_output, is_train):
    '''
    input_tensor,  # 数据入口
    demos,  # 模型资料（list）
    num_output,  # 出口数量
    is_train): # BN是否被训练
    '''
    data = input_tensor  # 第一层卷积7*7,stride = 2,深度为64
    data = conv2d_same(data, 64, 7, 2, is_train, None, normalizer_fn=False)
    data = slim.max_pool2d(data, 3, 2, scope="pool_1")
    with tf.variable_scope("resnet"):  # 堆叠总类瓶颈模块
        demo_num = 0
        for demo in demos:
            demo_num += 1
            # 堆叠子类瓶颈模块
            print("--------------------------------------------")
            for i in range(demo["num_class"]):
                print(demo_num)
                if demo_num is not 4:
                    if i == demo["num_class"] - 1:
                        stride = 2
                    else:
                        stride = 1
                else:
                    stride = 1
                data = bottleneck(data, demo["depth"], stride, is_train)
            print("--------------------------------------------")
    data = tf.layers.batch_normalization(data, training=is_train)
    data = tf.nn.relu(data)  # 平均池化，也可用Avg_pool函数
    data = tf.reduce_mean(data, [1, 2], keep_dims=True)
    print("output : ", data)  # 最后全连接层
    data = slim.conv2d(data, num_output, 1, activation_fn=None)
    data_shape = data.get_shape().as_list()
    nodes = data_shape[1] * data_shape[2] * data_shape[3]
    data = tf.reshape(data, [-1, nodes])
    return data


'''
inference(input_tensor = 数据入口
                        demos = ResNet_demo["layer_101"],         #获取模型词典
                        num_output = 出口数量,
                        is_train = False)      # BN是否被训练
'''
