
import numpy as np
import tensorflow as tf
import os, sys

PATH_DIRECTORY = os.path.abspath('../..')
PATH_TESTER_DIR = os.path.dirname('__file__')
PATH_SRC_DIR = '..'
sys.path.insert(0, os.path.abspath(os.path.join(PATH_TESTER_DIR, PATH_SRC_DIR)))

from tools.utils import load_image, show_image, print_prob_all
from nets.mlp import mlp as net_mlp


data = np.random.random([2,5])
print(data)

with tf.Session() as sess:
    # VARIABLES
    inputs = tf.placeholder('float', [None, 5])
    train_mode = tf.placeholder(tf.bool)

    # MODEL VGG19
    mlp = net_mlp(None, trainable=True, learning_rate=0.001)
    mlp.build(inputs, train_mode=train_mode)
    
#     # RUN TEST
    sess.run(tf.global_variables_initializer())
    for i in range(6000):
        if i%10 == 0:
            cost = sess.run(mlp.cost, feed_dict={inputs: data, train_mode: False})
            print(cost)
        else:
            sess.run(mlp.train, feed_dict={inputs: data, train_mode: True})
    
    inn, prob, cost = sess.run([mlp.input, mlp.prob, mlp.cost], feed_dict={inputs: data, train_mode: False})
    print(cost)
    print('INPUT', inn)
    print('OUTPUT', prob)

