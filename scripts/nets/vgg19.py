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

class cnn_vgg19:
    def __init__(self, npy_path=None, trainable=True, learning_rate=0.05, dropout=0.5, load_weight_fc=True):
        """
        load variable from npy to build the vgg
        :param npy_path: [string] path of the file with extension .npy
        :param trainable: [bool] auxiliary variable to build the graph 
        :param learning_rate: [float]
        :param dropout: [float]
        :param load_weight_fc: [bool] variable for transfer learning and fine-tunning, assigns random values ​​to the last fully connected layers 
        """

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
        self.save_data_external = {}
    
    def build(self, rgb, labels=[], train_mode=None, dim_last_layers=[4096, 4096, 1000]):
        """
        build the graph with tensorflow
        :param rgb: rgb image [#batch, height, width, 3] values scaled [0, 1]
        :param labels: labels of images [#batch]
        :param train_mode: a bool tensor, usually a placeholder: if True, dropout will be turned on
        :param dim_last_layers: size of the last layers classification by default vgg19 use 4096, 1000 class
        """
        assert len(dim_last_layers) == 3
        self.num_class = dim_last_layers[-1]

        start_time = time.time()
        print(PREPEND + "- build model started")
        self.inputOriginal = rgb
        rgb_scaled = rgb
        rgb_scaled = rgb * 255.0

        # Convert RGB to BGR
        red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=rgb_scaled)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        bgr = tf.concat(axis=3, values=[
            blue - VGG_MEAN[0],
            green - VGG_MEAN[1],
            red - VGG_MEAN[2],
        ])
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        self.input = bgr
        self.conv1_1 = self.conv_layer(self.input, 3, 64, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, 64, 128, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, 128, 128, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, 128, 256, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, 256, 256, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, 256, 256, "conv3_3")
        self.conv3_4 = self.conv_layer(self.conv3_3, 256, 256, "conv3_4")
        self.pool3 = self.max_pool(self.conv3_4, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, 256, 512, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, 512, 512, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, 512, 512, "conv4_3")
        self.conv4_4 = self.conv_layer(self.conv4_3, 512, 512, "conv4_4")
        self.pool4 = self.max_pool(self.conv4_4, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, 512, 512, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, 512, 512, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, 512, 512, "conv5_3")
        self.conv5_4 = self.conv_layer(self.conv5_3, 512, 512, "conv5_4")
        self.pool5 = self.max_pool(self.conv5_4, 'pool5')

        self.fc6 = self.fc_layer(self.pool5, 25088, dim_last_layers[0], "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
        self.relu6 = tf.nn.relu(self.fc6)

        if train_mode is not None:
            self.relu6 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu6, self.dropout), lambda: self.relu6)
        elif self.trainable:
            self.relu6 = tf.nn.dropout(self.relu6, self.dropout)

        self.fc7 = self.fc_layer(self.relu6, dim_last_layers[0], dim_last_layers[1], "fc7")
        self.relu7 = tf.nn.relu(self.fc7)
        if train_mode is not None:
            self.relu7 = tf.cond(train_mode, lambda: tf.nn.dropout(self.relu7, self.dropout), lambda: self.relu7)
        elif self.trainable:
            self.relu7 = tf.nn.dropout(self.relu7, self.dropout)

        self.fc8 = self.fc_layer(self.relu7, dim_last_layers[1], dim_last_layers[2], "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        # COST - TRAINING
        # if train_mode is not None and train_mode is True:
        if self.trainable:
            target = tf.cast(tf.one_hot(labels, on_value=1, off_value=0, depth=self.num_class), tf.float32)
            self.cost = tf.reduce_mean((self.prob - target) ** 2)
            self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)
            # tf.summary.scalar('cost_vs_batch', self.cost)
            # self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)

            # ACCURACY
            with tf.name_scope('accuracy'):
                with tf.name_scope('correct_prediction'):
                    self.labels_pred = tf.argmax(self.prob, 1)
                    self.correct_prediction = tf.equal(self.labels_pred, labels)
                    self.correct_count = tf.count_nonzero(self.correct_prediction)
                with tf.name_scope('accuracy'):
                    self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))
                    # tf.summary.scalar('accuracy_vs_batch', self.accuracy)

        # self.merged = tf.summary.merge_all()
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
        assert var.get_shape() == initial_value.get_shape(), 'The layer ['+name+'] '+ str(var.get_shape())+' !== '+ str(initial_value.get_shape()) 
        return var
    
    def save_npy(self, sess, npy_path="./vgg19-save.npy"):
        assert isinstance(sess, tf.Session)

        paths = npy_path.split('/')
        paths_npy = '/'.join(paths[:-1])
        assert os.path.exists(paths_npy), 'Save npy: The directory do not exist.'

        data_dict = {}

        for (name, idx), var in list(self.var_dict.items()):
            var_out = sess.run(var)
            if name not in data_dict:
                data_dict[name] = {}
            data_dict[name][idx] = var_out

        np.save(npy_path, data_dict)
        print(PREPEND + "- File saved", npy_path)
        return npy_path
    
    def add_row_data_by_save(self, data, labels, index):
        self.save_data_external[index] = {}
        self.save_data_external[index][0] = data
        self.save_data_external[index][1] = labels

    def save_data_npy(self, sess, npy_path="./data.npy"):
        assert isinstance(sess, tf.Session)
        paths = npy_path.split('/')
        paths_npy = '/'.join(paths[:-1])
        assert os.path.exists(paths_npy), 'Save data npy: The directory do not exist.'

        data_dict = {}
        data_dict['data'] = self.save_data_external
        np.save(npy_path, data_dict)
        print(PREPEND + "- File Data saved", npy_path)
        return npy_path

    def get_var_count(self):
        count = 0
        for v in list(self.var_dict.values()):
            count += reduce(lambda x, y: x * y, v.get_shape().as_list())
        return count
