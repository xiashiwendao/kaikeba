import tensorflow as tf
from tensorflow.python.ops import array_ops

def binary_focal_loss(gamma=2, alpha=0.25):
    alpha = tf.constant(alpha, dtype=tf.float32)
    gamma = tf.constant(gamma, dtype=tf.float32)
    def binary_focal_loss_fixed(n_classes, logits, true_label):
        epsilon = 1.e-8
        # 得到y_true和y_pred
        y_true = tf.one_hot(true_label, n_classes)
        probs = tf.nn.sigmoid(logits)
        y_pred = tf.clip_by_value(probs, epsilon, 1. - epsilon)
        # 得到调节因子weight和alpha
        ## 先得到y_true和1-y_true的概率【这里是正负样本的概率都要计算哦！】
        p_t = y_true * y_pred \
              + (tf.ones_like(y_true) - y_true) * (tf.ones_like(y_true) - y_pred)
        ## 然后通过p_t和gamma得到weight
        weight = tf.pow((tf.ones_like(y_true) - p_t), gamma)
        ## 再得到alpha，y_true的是alpha，那么1-y_true的是1-alpha
        alpha_t = y_true * alpha + (tf.ones_like(y_true) - y_true) * (1 - alpha)
        # 最后就是论文中的公式，相当于：- alpha * (1-p_t)^gamma * log(p_t)
        focal_loss = - alpha_t * weight * tf.log(p_t)
        return tf.reduce_mean(focal_loss)

def focal_loss(logits, labels, gamma):
    '''
    :param logits:  [batch_size, n_class]
    :param labels: [batch_size]
    :return: -(1-y)^r * log(y)
    '''
    softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
    labels = tf.range(0, logits.shape[0]) * logits.shape[1] + labels
    prob = tf.gather(softmax, labels)
    weight = tf.pow(tf.subtract(1., prob), gamma)
    loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))

    return loss   

def focal_loss(batch_size, n_class, logits, labels, alpha, epsilon = 1.e-7, gamma=2.0, multi_dim = False):
        '''
        :param logits:  [batch_size, n_class]
        :param labels: [batch_size]  not one-hot !!!
        :return: -alpha*(1-y)^r * log(y)
        它是在哪实现 1- y 的？ 通过gather选择的就是1-p,而不是通过计算实现的；
        logits soft max之后是多个类别的概率，也就是二分类时候的1-P和P；多分类的时候不是1-p了；

        怎么把alpha的权重加上去？
        通过gather把alpha选择后变成batch长度，同时达到了选择和维度变换的目的

        是否需要对logits转换后的概率值进行限制？
        需要的，避免极端情况的影响

        针对输入是 (N，P，C )和  (N，P)怎么处理？
        先把他转换为和常规的一样形状，（N*P，C） 和 （N*P,）

        bug:
        ValueError: Cannot convert an unknown Dimension to a Tensor: ?
        因为输入的尺寸有时是未知的，导致了该bug,如果batchsize是确定的，可以直接修改为batchsize

        '''

        if multi_dim:
            logits = tf.reshape(logits, [-1, logits.shape[2]])
            labels = tf.reshape(labels, [-1])

        # (Class ,1)
        alpha = tf.constant(alpha, dtype=tf.float32)

        labels = tf.cast(labels, dtype=tf.int32)
        logits = tf.cast(logits, tf.float32)
        # (N,Class) > N*Class
        softmax = tf.reshape(tf.nn.softmax(logits), [-1])  # [batch_size * n_class]
        # (N,) > (N,) ,但是数值变换了，变成了每个label在N*Class中的位置
        # labels_shift = tf.range(0, logits_raw[0]) * logits_raw[1] + labels
        labels_shift = tf.range(0, batch_size) * n_class + labels
        # (N*Class,) > (N,)
        prob = tf.gather(softmax, labels_shift)
        # 预防预测概率值为0的情况  ; (N,)
        prob = tf.clip_by_value(prob, epsilon, 1. - epsilon)
        # (Class ,1) > (N,)
        alpha_choice = tf.gather(alpha, labels)
        # (N,) > (N,)
        weight = tf.pow(tf.subtract(1., prob), gamma)
        weight = tf.multiply(alpha_choice, weight)
        # (N,) > 1
        loss = -tf.reduce_mean(tf.multiply(weight, tf.log(prob)))
        return loss


def focal_loss_v2(prediction_tensor, target_tensor, weights=None, alpha=0.25, gamma=2):
    r"""Compute focal loss for predictions.
        Multi-labels Focal loss formula:
            FL = -alpha * (z-p)^gamma * log(p) -(1-alpha) * p^gamma * log(1-p)
                 ,which alpha = 0.25, gamma = 2, p = sigmoid(x), z = target_tensor.
    Args:
     prediction_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing the predicted logits for each class
     target_tensor: A float tensor of shape [batch_size, num_anchors,
        num_classes] representing one-hot encoded classification targets
     weights: A float tensor of shape [batch_size, num_anchors]
     alpha: A scalar tensor for focal loss alpha hyper-parameter
     gamma: A scalar tensor for focal loss gamma hyper-parameter
    Returns:
        loss: A (scalar) tensor representing the value of the loss function
    """
    sigmoid_p = tf.nn.sigmoid(prediction_tensor)
    zeros = array_ops.zeros_like(sigmoid_p, dtype=sigmoid_p.dtype)
    
    # For poitive prediction, only need consider front part loss, back part is 0;
    # target_tensor > zeros <=> z=1, so poitive coefficient = z - p.
    pos_p_sub = array_ops.where(target_tensor > zeros, target_tensor - sigmoid_p, zeros)
    
    # For negative prediction, only need consider back part loss, front part is 0;
    # target_tensor > zeros <=> z=1, so negative coefficient = 0.
    neg_p_sub = array_ops.where(target_tensor > zeros, zeros, sigmoid_p)
    per_entry_cross_ent = - alpha * (pos_p_sub ** gamma) * tf.log(tf.clip_by_value(sigmoid_p, 1e-8, 1.0)) \
                          - (1 - alpha) * (neg_p_sub ** gamma) * tf.log(tf.clip_by_value(1.0 - sigmoid_p, 1e-8, 1.0))
    return tf.reduce_sum(per_entry_cross_ent)


def focal_loss_sigmoid(labels,logits,alpha=0.25,gamma=2):
    """
    Computer focal loss for binary classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size].
      alpha: A scalar for focal loss alpha hyper-parameter. If positive samples number
      > negtive samples number, alpha < 0.5 and vice versa.
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.sigmoid(logits)
    labels=tf.to_float(labels)
    L=-labels*(1-alpha)*((1-y_pred)*gamma)*tf.log(y_pred)-\
      (1-labels)*alpha*(y_pred**gamma)*tf.log(1-y_pred)
    return L

def focal_loss_softmax(labels,logits,class_num, gamma=2):
    """
    Computer focal loss for multi classification
    Args:
      labels: A int32 tensor of shape [batch_size].
      logits: A float32 tensor of shape [batch_size,num_classes].
      gamma: A scalar for focal loss gamma hyper-parameter.
    Returns:
      A tensor of the same shape as `lables`
    """
    y_pred=tf.nn.softmax(logits,dim=-1) # [batch_size,num_classes]
    # labels=tf.one_hot(labels,depth=y_pred.shape[1])
    labels=tf.one_hot(labels,depth=class_num)
    L=-labels*((1-y_pred)**gamma)*tf.log(y_pred)
    L=tf.reduce_sum(L,axis=1)
    return L

if __name__ == '__main__':
    logits=tf.random_uniform(shape=[5],minval=-1,maxval=1,dtype=tf.float32)
    labels=tf.Variable([0,1,0,0,1])
    loss1=focal_loss_sigmoid(labels=labels,logits=logits)

    logits2=tf.random_uniform(shape=[5,4],minval=-1,maxval=1,dtype=tf.float32)
    labels2=tf.Variable([1,0,2,3,1])
    loss2=focal_loss_softmax(labels==labels2,logits=logits2)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        print(sess.run(loss1))
        print(sess.run(loss2))

