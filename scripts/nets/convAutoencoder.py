import tensorflow as tf
import time
import numpy as np

class ConvAEncoder:

    def __init__(self, npy_path=None, trainable=True, learning_rate=0.001, dropout=0.5):
        if npy_path is not None:
            self.data_dict = np.load(npy_path, encoding='latin1').item()
            print("npy file loaded")
        else:
            self.data_dict = None
            print("random weight")

        self.var_dict = {}
        self.trainable = trainable
        self.learning_rate = learning_rate
        self.dropout = dropout
        self.encoder_w = []
        self.net = {}

    def build(self, input_batch, n_filters=[1, 10, 10, 10], corruption=False):

        start_time = time.time()

        #
        # 2-D is CONVERTED TO SQUARE TENSOR.
        # ---------------------------------

        if len(input_batch.get_shape()) == 2:
            x_dim = np.sqrt(input_batch.get_shape().as_list()[1])
            if x_dim != int(x_dim):
                raise ValueError('Unsupported input dimensions')
            x_dim = int(x_dim)
            x_tensor = tf.reshape(input_batch, [-1, x_dim, x_dim, n_filters[0]])
        elif len(input_batch.get_shape()) == 4:
            x_tensor = input_batch
        else:
            raise ValueError('Unsupported input dimensions')
        current_input = x_tensor

        #
        # OPTIONALLY APPLY DENOISING AUTOENCODER
        # --------------------------------------

        if corruption is True:
            current_input = self.corrupt(current_input)

        #
        # BUILD THE ENCODER
        # -----------------
        self.x = current_input
        shapes = []
        for i, n_output in enumerate(n_filters[1:]):

            shapes.append(current_input.get_shape().as_list())
            n_input = current_input.get_shape().as_list()[3]
            name = 'encodeConv_' + str(i)
            self.net[name] = self.conv_layer(current_input, n_input, n_output, name)
            current_input = self.net[name]

        #
        # BUILD THE DECODER USING THE SAME WEIGHTS
        # ----------------------------------------

        self.z = current_input
        self.encoder_w.reverse()
        shapes.reverse()

        for i, shape in enumerate(shapes):
            name = 'decodeConv_' + str(i)
            self.net[name] = self.conv_trans_layer(current_input, shape, i, name)
            current_input = self.net[name]

        self.y = current_input
        self.cost = tf.reduce_sum(tf.square(self.y - x_tensor))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        self.encoder_w = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def conv_layer(self, bottom, in_channels, out_channels, name):
        with tf.variable_scope(name):
            filt, conv_biases = self.get_conv_var(3, in_channels, out_channels, name, type=1)

            conv = tf.nn.conv2d(bottom, filt, [1, 2, 2, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)

            return relu

    def conv_trans_layer(self, bottom, shape, index, name):
        with tf.variable_scope(name):
            initial_value = self.encoder_w[index]
            filt = self.get_var(initial_value, name, 0, name + "_filters")

            initial_value = tf.truncated_normal([filt.get_shape().as_list()[2]], .0, .001)
            # initial_value = tf.zeros([filt.get_shape().as_list()[2]])
            conv_biases = self.get_var(initial_value, name, 1, name + "_biases")

            conv = tf.nn.conv2d_transpose(bottom, filt, tf.stack([tf.shape(bottom)[0], shape[1], shape[2], shape[3]]), strides=[1, 2, 2, 1], padding='SAME')
            bias = tf.nn.bias_add(conv, conv_biases)
            relu = tf.nn.relu(bias)
            return relu

    def get_conv_var(self, filter_size, in_channels, out_channels, name, type=1):

        if type == 1:
            initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.1)
            filters = self.get_var(initial_value, name, 0, name + "_filters")
            self.encoder_w.append(initial_value)

            initial_value = tf.truncated_normal([out_channels], .0, .1)
            biases = self.get_var(initial_value, name, 1, name + "_biases")
        
        elif type == 2:
            initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], 0.0, 0.001)
            filters = self.get_var(initial_value, name, 0, name + "_filters")
            self.encoder_w.append(initial_value)

            initial_value = tf.truncated_normal([out_channels], .0, .001)
            biases = self.get_var(initial_value, name, 1, name + "_biases")

        elif type == 3:
            mm = np.sqrt(in_channels)
            initial_value = tf.truncated_normal([filter_size, filter_size, in_channels, out_channels], -1.0/mm, 1.0/mm)
            filters = self.get_var(initial_value, name, 0, name + "_filters")
            self.encoder_w.append(initial_value)
            initial_value = tf.zeros([out_channels])
            biases = self.get_var(initial_value, name, 1, name + "_biases")

        return filters, biases

    def get_var(self, initial_value, name, idx, var_name):

        if self.data_dict is not None and name in self.data_dict:
            value = self.data_dict[name][idx]
        else:
            value = initial_value

        if self.trainable:
            var = tf.Variable(value, name=var_name)
        else:
            var = tf.constant(value, dtype=tf.float32, name=var_name)

        self.var_dict[(name, idx)] = var
        assert var.get_shape() == initial_value.get_shape()
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
        print("File saved", npy_path)
        return npy_path

    def corrupt(self, x):
        # mask_np = np.random.binomial(1, 1 - self.noise, tf.shape(x))
        mask_np = tf.cast(tf.random_uniform(shape=tf.shape(x), minval=0, maxval=2, dtype=tf.int32), tf.float32)
        return tf.multiply(x, mask_np)
