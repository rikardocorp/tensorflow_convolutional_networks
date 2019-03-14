import tensorflow as tf
import time
import numpy as np


class AEncoder:
    def __init__(self, npy_path=None, trainable=True, learning_rate=0.001, dropout=0.5, noise=0.0):
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
        self.noise = noise

    def build(self, input_batch, input_mask, noise_mode=False, l_hidden=[2048, 1024]):

        start_time = time.time()
        # current_input = input_batch * input_mask
        current_input = tf.cond(noise_mode, lambda: input_batch * input_mask, lambda: input_batch)

        #
        # BUILD THE ENCODER
        # -----------------
        shapes = []
        self.x = current_input
        for i, n_output in enumerate(l_hidden[0:]):
            shapes.append(current_input.get_shape().as_list())
            n_input = current_input.get_shape().as_list()[1]
            name = 'encodeFC_' + str(i)

            self.net[name] = self.fc_layer_sigm(current_input, n_input, n_output, name)
            current_input = self.net[name]

        #
        # BUILD THE DECODER USING THE SAME WEIGHTS
        # ----------------------------------------
        self.z = current_input
        self.encoder_w.reverse()
        shapes.reverse()

        for i, shape in enumerate(shapes):
            name = 'decodeFC_' + str(i)
            n_input = current_input.get_shape().as_list()[1]
            self.net[name] = self.fc_layer_sigm_decode(current_input, n_input, shape[1], i, name)
            current_input = self.net[name]

        self.y = current_input
        self.cost = tf.reduce_sum(tf.square(self.y - input_batch))
        self.train = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        # self.train = tf.train.GradientDescentOptimizer(self.learning_rate).minimize(self.cost)

        self.data_dict = None
        self.encoder_w = None
        print(("build model finished: %ds" % (time.time() - start_time)))

    def fc_layer_sigm(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name, type=2)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
            return fc

    def fc_layer_sigm_decode(self, bottom, in_size, out_size, index, name):
        with tf.variable_scope(name):
            initial_value = tf.transpose(self.encoder_w[index])
            weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

            # initial_value = tf.truncated_normal([out_size], .0, .001)
            initial_value = tf.zeros([out_size])
            biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.sigmoid(tf.matmul(x, weights) + biases)
            return fc

    def fc_layer(self, bottom, in_size, out_size, name):
        with tf.variable_scope(name):
            weights, biases = self.get_fc_var(in_size, out_size, name)

            x = tf.reshape(bottom, [-1, in_size])
            fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
            return fc

    def get_fc_var(self, in_size, out_size, name, type=1):

        if type == 1:
            initial_value = tf.truncated_normal([in_size, out_size], 0.0, 0.001)
            self.encoder_w.append(initial_value)
            weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

            initial_value = tf.truncated_normal([out_size], .0, .001)
            biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        elif type == 2:
            w_init_max = 4 * np.sqrt(6. / (in_size + out_size))
            initial_value = tf.random_uniform([in_size, out_size], minval=-w_init_max, maxval=w_init_max)
            self.encoder_w.append(initial_value)
            weights = self.get_var_fc(initial_value, name, 0, name + "_weights")

            initial_value = tf.zeros([out_size])
            biases = self.get_var_fc(initial_value, name, 1, name + "_biases")

        return weights, biases

    def get_var_fc(self, initial_value, name, idx, var_name):

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
