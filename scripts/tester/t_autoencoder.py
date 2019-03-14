import matplotlib
matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
import os
import sys
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

PATH_DIRECTORY = os.path.abspath('../..')
PATH_TESTER_DIR = os.path.dirname('__file__')
PATH_SRC_DIR = '..'
sys.path.insert(0, os.path.abspath(os.path.join(PATH_TESTER_DIR, PATH_SRC_DIR)))


from nets.autoencoder_noise import AEncoder
from tools.utils import load_image, show_image, print_prob_all


mnist = input_data.read_data_sets(PATH_DIRECTORY + "/data/MNIST_data/", one_hot=False)


def plot_images(images_x, images_y, cls_true, shape=(28,28)):
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.5, wspace=0.5)

    for i, ax in enumerate(axes.flat):
        new_image = np.hstack((images_x[i].reshape(shape), images_y[i].reshape(shape)))
        ax.imshow(new_image, cmap='binary')

        # Show true and predicted classes.
        xlabel = "True: {0}".format(cls_true[i])
        ax.set_xlabel(xlabel)

        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

def feed_dict(train, noise=False, noise_value=0):
    """Make a TensorFlow feed_dict: maps data onto Tensor placeholders."""

    if train:
        xs, ys = mnist.train.next_batch(100)
    else:
        xs, ys = mnist.test.images, mnist.test.labels
    
    if noise:
        mask_np = np.random.binomial(1, 1 - noise_value, xs.shape)
    else:
        mask_np = np.random.random([0, xs.shape[1]])

    return {x_batch: xs, mask: mask_np, noise_mode: noise}


def test_model(net, sess_test):
    print('\n# PHASE: Testing model')
    cost, x, y = sess_test.run([net.cost, net.x, net.y], feed_dict=feed_dict(False, noise=False))
    print('     Cost:', cost)
    label = mnist.test.labels
    return x, y, label

def train_model(net, sess_train, epoch):
    print('\n# PHASE: Training model')
    for ep in range(epoch):
        _, cost = sess_train.run([net.train, net.cost], feed_dict=feed_dict(True, noise=False))
        print('     Epoch:', ep, cost)


# # data = np.array([[1,0,0,0,0,0,1,0,0,1], [1,0,0,1,0,0,0,0,0,1]])
# data = np.random.random([10, 10])
# print(data)
learning_rate = 0.0005
noise_level = 0
epoch = 100

path_load_weight = PATH_DIRECTORY + '/weights/aencoder/ae.npy'
path_save_weight = PATH_DIRECTORY + '/weights/aencoder/ae.npy'

with tf.Session() as sess:
    x_batch = tf.placeholder(tf.float32, [None, 784])
    mask = tf.placeholder(tf.float32, [None, 784])
    noise_mode = tf.placeholder(tf.bool)

    AEncode = AEncoder(None, learning_rate=learning_rate, noise=noise_level)
    AEncode.build(x_batch, mask, noise_mode, [500, 250])
    sess.run(tf.global_variables_initializer())

    _x, _y, labels = test_model(AEncode, sess)
    plot_images(_x[:9,:], _y[:9,:], labels)

    for i in range(15):
        train_model(AEncode, sess, epoch)
        _x, _y, labels = test_model(AEncode, sess)
        plot_images(_x[:9,:], _y[:9,:], labels)
    
    # AEncode.save_npy(sess, PATH_DIRECTORY + '/weights/aencoder/ae.npy')