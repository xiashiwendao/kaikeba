import tensorflow as tf
from inputdata import read_data_sets
mnist = read_data_sets("MNIST_data/", one_hot=True)

# 标准差为0.1的正态分布


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# 0.1的偏差常数，为了避免死亡节点


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 二维卷积函数
# strides代表卷积模板移动的步长，全是1代表走过所有的值
# padding设为SAME意思是保持输入输出的大小一样，使用全0补充


def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

# ksize [1, height, width, 1] 第一个和最后一个代表对batches和channel做池化，1代表不池化
# strides [1, stride,stride, 1]意思是步长为2，我们使用的最大池化


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])
# reshape图片成28 * 28 大小，-1代表样本数量不固定，1代表channel
x_image = tf.reshape(x, [-1, 28, 28, 1])

# 前面两个5代表卷积核的尺寸，1代表channel，32代表深度
W_conv1 = weight_variable([5, 5, 1, 32])
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1)+b_conv1)
# p_conv1 返回的是第一层的返回值，是第二层的输入值（h_conv2将会使用到p_conv1）
p_conv1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, 32, 64])
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(p_conv1, W_conv2)+b_conv2)
p_conv2 = max_pool_2x2(h_conv2)

# 输入为7 * 7 * 64， 输出为1024的一维向量（权重）
W_fc1 = weight_variable([7*7*64, 1024])
b_fc1 = bias_variable([1024])
# 上面指定了卷积维度，这里将待卷积矩阵reshape为可以和卷积工作的工作
h_pool2_flat = tf.reshape(p_conv2, [-1, 7*7*64])
# 全连接这里不再使用卷积，而是采用矩阵相乘
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1)+b_fc1)

# dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# softmax
# 注意，因为手写体识别要识别10个数字，所以第二个维度是10，第一个维度1024是提取的特征图
W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# 交叉熵
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ *
                                              tf.log(y_conv), reduction_indices=[1]))
# 参数优化器Adam代替梯度递减，当然都可以使用，这里只是用一个和以前不一样的
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# 正确的预测结果
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
# 计算预测准确率
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 训练
# tf.global_variables_initilizer().run()
sess = tf.Session()
tf.global_variables_initializer().run(session=sess)
for i in range(2000):
    batch = mnist.train.next_batch(50)
    # 每一百次打印一次结果,keep_prob设为1意思是用原数据进行检测，所以保留所有数据
    if(i % 100 == 0):
        # placeholder的值都是在这里被指定的，eval其实就是run；eval和spark里面的action算子是一样的，调用才会触发图的执行
        # 之前的算子（conv2d，reduce_mean等都是tranformation）
        train_acuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0}, session=sess)
        print("step %d, tranning accuracy %g:" % (i, train_acuracy))

    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5}, session=sess)

eval_val = accuracy.eval(feed_dict={
                         x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}, session=sess)
# 训练结束后，在测试集上进行测试
# print("Model accuracy: ", accuracy.eval(feed.dict={x:mnist.test.images,y_:mnist.test.labels,keep_prob:1.0}))
print("Model accuracy: ", eval_val)
