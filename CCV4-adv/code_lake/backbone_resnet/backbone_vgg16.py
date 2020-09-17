# -*- coding:utf-8 -*-
#
# VGG-16 Net model
 
import numpy as np
import tensorflow as tf
 
class Vgg16:
    def __init__(self, images, name):
        self.name = name
        self.input = images
        self.output = self.vgg16(self.input)
        list_vars = tf.trainable_variable()
        self.vars = [var for var in list_vars]
        
    def get_conv_weight(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
 
    def get_bias(self, shape, name):
        return tf.Variable(tf.constant(0.0, shape=shape), name=name)
 
    def get_fc_weight(self, shape, name):
        return tf.Variable(tf.truncated_normal(shape, stddev=0.1), name=name)
    
    def conv_layer(self, x, ks, out_units, name):
        with tf.variable_scope(name):
            in_units = x.get_shape().as_list()[-1]
            filt = self.get_conv_weight([ks,ks,in_units,out_units], name='weight')
            bias = self.get_bias([out_units], name='bias')
            out  = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x, filt, [1,1,1,1], padding='SAME'), bias))
            return out
 
    def fc_layer(self, x, out_units, name):
        with tf.variable_scope(name):
            in_units = np.prod(x.get_shape().as_list()[1:])
            x_flat = tf.reshape(x, [-1, in_units])
            weight = self.get_fc_weight([in_units,out_units], name='weight')
            biases = self.get_bias([out_units], name='bias')
            out    = tf.nn.bias_add(tf.matmul(x_flat, weight), biases)
            return out
 
    def avg_pool(self, x, name):
        return tf.nn.avg_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
 
    def max_pool(self, x, name):

        return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME', name=name)
 
    def vgg16(self, x, keep_prob):
        conv1_1 = self.conv_layer(x, ks=3, out_units=64, name='conv1_1')
        conv1_2 = self.conv_layer(conv1_1, ks=3, out_units=64, name='conv1_2')
        pool1   = self.max_pool(conv1_2, 'pool1')
 
        conv2_1 = self.conv_layer(pool1, ks=3, out_units=128, name='conv2_1')
        conv2_2 = self.conv_layer(conv2_1, ks=3, out_units=128, name='conv2_2')
        pool2   = self.max_pool(conv2_2, 'pool2')
 
        conv3_1 = self.conv_layer(pool2, ks=3, out_units=256, name='conv3_1')
        conv3_2 = self.conv_layer(conv3_1, ks=3, out_units=256, name='conv3_2')
        conv3_3 = self.conv_layer(conv3_2, ks=3, out_units=256, name='conv3_3')
        pool3   = self.max_pool(conv3_3, 'pool3')
 
        conv4_1 = self.conv_layer(pool3, ks=3, out_units=512, name='conv4_1')
        conv4_2 = self.conv_layer(conv4_1, ks=3, out_units=512, name='conv4_2')
        conv4_3 = self.conv_layer(conv4_2, ks=3, out_units=512, name='conv4_3')
        pool4   = self.max_pool(conv4_3, 'pool4')
 
        conv5_1 = self.conv_layer(pool4, ks=3, out_units=512, name='conv5_1')
        conv5_2 = self.conv_layer(conv5_1, ks=3, out_units=512, name='conv5_2')
        conv5_3 = self.conv_layer(conv5_2, ks=3, out_units=512, name='conv5_3')
        pool5   = self.max_pool(conv5_3, 'pool5')
 
        fc6      = self.fc_layer(pool5, out_units=4096, name='fc6')
        fc6_relu = tf.nn.relu(fc6)
        fc6_drop = tf.nn.dropout(fc6_relu, keep_prob, name='fc6_drop')
 
        fc7      = self.fc_layer(fc6_drop, name='fc7')
        fc7_relu = tf.nn.relu(fc7)
        fc7_drop = tf.nn.dropout(fc7_relu, keep_prob, name='fc7_drop')
 
        fc8 = self.fc_layer(fc7_drop, 'fc8')
        out = tf.nn.softmax(fc8, name='out')
        return out