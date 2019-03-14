
import numpy as np
import tensorflow as tf
import os, sys

PATH_DIRECTORY = os.path.abspath('../..')
PATH_TESTER_DIR = os.path.dirname('__file__')
PATH_SRC_DIR = '..'
sys.path.insert(0, os.path.abspath(os.path.join(PATH_TESTER_DIR, PATH_SRC_DIR)))

from tools.utils import load_image, show_image, print_prob_all
from nets.vgg19 import cnn_vgg19

img1 = load_image(PATH_DIRECTORY + '/data/test/tigerTrain1.jpeg')
img2 = load_image(PATH_DIRECTORY + '/data/test/tigerTrain2.jpeg')
img3 = load_image(PATH_DIRECTORY + '/data/test/tigerTrain3.jpg')
img4 = load_image(PATH_DIRECTORY + '/data/test/tigerTest1.jpg')
img5 = load_image(PATH_DIRECTORY + '/data/test/tigerTest2.jpg')

batch1 = img1.reshape((1,224,224,3))
batch2 = img2.reshape((1,224,224,3))
batch3 = img3.reshape((1,224,224,3))
batch4 = img4.reshape((1,224,224,3))
batch5 = img5.reshape((1,224,224,3))

batch_train = np.concatenate((batch1, batch2, batch3), 0)
label = [292, 292, 292]
batch_test = np.concatenate((batch4, batch5), 0)

batch = img1.reshape((1, 224, 224, 3))

npy_path = PATH_DIRECTORY + '/weights/vgg19/vgg19.npy'
data_label_path = PATH_DIRECTORY + '/data/synset.txt'

with tf.Session() as sess:
    # VARIABLES
    images = tf.placeholder(tf.float32, [None, 224, 224, 3])
    labels = tf.placeholder(tf.int64, [None])
    train_mode = tf.placeholder(tf.bool)

    # MODEL VGG19
    vgg19 = cnn_vgg19(npy_path, trainable=True)
    vgg19.build(images, labels, train_mode)

    # print number of variables used: 143667240 variables, i.e. ideal size = 548MB
    print('     # Number of variables:',vgg19.get_var_count())
    sess.run(tf.global_variables_initializer())

    # RUN TEST #1
    prob = sess.run(vgg19.prob, feed_dict={images: batch_test, labels: label, train_mode: False})
    print_prob_all(prob, data_label_path)

    #RUN TRAIN
    print('     - training...')
    sess.run(vgg19.train, feed_dict={images: batch_train, labels: label, train_mode: True})
    sess.run(vgg19.train, feed_dict={images: batch_train, labels: label, train_mode: True})
    sess.run(vgg19.train, feed_dict={images: batch_train, labels: label, train_mode: True})
    print('')

    # RUN TEST #2
    prob = sess.run(vgg19.prob, feed_dict={images: batch, labels: label, train_mode: False})
    print_prob_all(prob, data_label_path)