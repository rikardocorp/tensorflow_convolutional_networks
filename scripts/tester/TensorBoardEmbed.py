# https://github.com/anujshah1003/Tensorboard-examples/blob/master/mnist-tensorboard-embeddings-1.py

import os, sys
import tensorflow as tf

try:
    from tensorflow.examples.tutorials.mnist import input_data
    from tensorflow.contrib.tensorboard.plugins import projector
except:
    pass
import numpy as np

PATH_DIRECTORY = os.path.abspath('../..')
PATH_TESTER_DIR = os.path.dirname('__file__')
PATH_SRC_DIR = '..'
sys.path.insert(0, os.path.abspath(os.path.join(PATH_TESTER_DIR, PATH_SRC_DIR)))

from tools.dataset import Dataset
from tools.utils import load_data_npy

LOG_DIR = PATH_DIRECTORY + '/graphs/log-1'
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

mnist = input_data.read_data_sets(PATH_DIRECTORY + "/data/MNIST_data/", one_hot=False)
images = tf.Variable(mnist.test.images, name='images')

with open(metadata, 'w') as metadata_file:
    for row in range(len(mnist.test.labels)):
        c = mnist.test.labels[row]
        metadata_file.write('{}\n'.format(c))

with tf.Session() as sess:

    # summary = sess.run(merged, feed_dict={x: data_final})
    # writer = tf.summary.FileWriter(LOG_DIR, sess.graph)
    # writer.add_summary(summary)

    saver = tf.train.Saver([images])
    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    # One can add multiple embeddings.
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    # Link this tensor to its metadata file (e.g. labels).
    embedding.metadata_path = metadata
    # Saves a config file that TensorBoard will read during startup.
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)

# for calling the tensorboard you should be in that drive and call the entire path
#tensorboard --logdir=/Technical_works/tensorflow/mnist-tensorboard/log-1 --port=6006