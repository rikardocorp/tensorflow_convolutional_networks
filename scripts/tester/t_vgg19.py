
import numpy as np
import tensorflow as tf
import os, sys

PATH_DIRECTORY = os.path.abspath('../..')
PATH_TESTER_DIR = os.path.dirname('__file__')
PATH_SRC_DIR = '..'
sys.path.insert(0, os.path.abspath(os.path.join(PATH_TESTER_DIR, PATH_SRC_DIR)))

from tools.utils import load_image, show_image, print_prob_all
from nets.vgg19 import cnn_vgg19

img1 = load_image(PATH_DIRECTORY + '/data/example/avion.jpeg')
img2 = load_image(PATH_DIRECTORY + '/data/example/tiger.jpeg')
batch1 = img1.reshape((1, 224, 224, 3))
batch2 = img2.reshape((1, 224, 224, 3))
batch = np.concatenate((batch1, batch2), 0)

npy_path = PATH_DIRECTORY + '/weights/vgg19/vgg19.npy'
data_label_path = PATH_DIRECTORY + '/data/synset.txt'

with tf.Session() as sess:
    # VARIABLES
    images = tf.placeholder(tf.float32, [2, 224, 224, 3])
    train_mode = tf.placeholder(tf.bool)

    # MODEL VGG19
    vgg19 = cnn_vgg19(npy_path, trainable=False)
    vgg19.build(images, train_mode=train_mode)
    
    # RUN TEST
    sess.run(tf.global_variables_initializer())
    prob, inputImage, inputOriginal = sess.run([vgg19.prob, vgg19.input, vgg19.inputOriginal], feed_dict={images: batch, train_mode: False})

    # show_image(inputOriginal[1])
    # show_image(inputImage[1])
    print_prob_all(prob, data_label_path, top=0)
