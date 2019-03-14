import os
import tensorflow as tf
import numpy as np
from functools import reduce
import time
import inspect

# VGG_MEAN (B, G, R )
VGG_MEAN = [103.939, 116.779, 123.68]
PREPEND = '     '
# print(VGG_MEAN)

class mlp:
    def __init__(self, npy_path=None, trainable=True, learning_rate=0.05, dropout=0.5, load_weight_fc=True):
        print('')
        print(PREPEND + '[INIT CNN_VGG19]')

        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print(PREPEND + "- npy file loaded")
        else:
            self.data_dict = None
            print(PREPEND + "- random weight")
        
        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.load_weight_fc = load_weight_fc
    
    def build(self, input, train_mode=None, dim_last_layers=[4096, 4096, 1000]):
        """
        load variable from npy to build the vgg
        :param rgb: rgb image [batch, height, width, 3] values scaled [0, 1]
        :param target: label image [#clases]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        :param dim_last_layers: size of the last layers classification by default vgg19 use 4096, 1000 class
        """
        start_time = time.time()
        self.input = input
        self.fc1 = self.fc_layer(self.input, 5, 10, "fc1")
        self.relu1 = tf.nn.relu(self.fc1)

        self.fc2 = self.fc_layer(self.relu1, 10, 5, "fc2")
        self.relu2 = tf.nn.relu(self.fc2)

        self.fc3 = self.fc_layer(self.relu2, 5, 10, "fc3")
        self.relu3 = tf.nn.relu(self.fc3)

        self.fc4 = self.fc_layer(self.relu3, 10, 5, "fc4")

        # self.prob = tf.nn.softmax(self.fc4, name="prob")
        self.prob = self.fc4

        # COST - TRAINING
        # if train_mode is not None and train_mode is True:
        if self.trainable:
            self.cost = tf.reduce_mean((self.prob - self.input) ** 2)
            self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            # self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        print((PREPEND + "- build model finished: %ds" % (time.time() - start_time)))
    
    def avg_pool(self, bottom, name):
        return tf.nn.avg_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)
    
    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name)
            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu
    
    def fc_layer(self, bottom, in_size, out_size, name, force_load_fc=False):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, force_load_fc)
            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc
    
    def get_conv_var(self, filter_size, in_channels, out_channels, name):
        initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
        filters = self.get_var(initial_value, name, 0, name + "_filters")

        initial_value = tf.truncated_normal([out_channels], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases")
        return filters, biases
    
    def get_fc_var(self, in_size, out_size, name, force_load_fc=False):
        initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
        weights = self.get_var(initial_value, name, 0, name + "_weights", is_convol=False, force_load_fc=force_load_fc) 

        initial_value = tf.truncated_normal([out_size], .0, .001)
        biases = self.get_var(initial_value, name, 1, name + "_biases", is_convol=False, force_load_fc=force_load_fc)
        return weights, biases
    
    def get_var(self, initial_value, name, idx, var_name, is_convol=True, force_load_fc=False):

        conditional = self.data_dict is not None and name in self.data_dict
        if not is_convol:
            conditional = conditional and ((self.load_weight_fc is True) or (force_load_fc is True))

        if conditional:
            value = self.data_dict[name][idx]
        else:
            value = initial_value
        
        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        # print(name, var.get_shape())
        # print var_name, var.get_shape().as_list()
        # assert var.get_shape() == initial_value.get_shape()
        return var
    
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(PREPEND + "- File saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
