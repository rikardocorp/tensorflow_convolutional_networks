import numpy as np
import tensorflow as tf
import os, sys
from os import listdir
from os.path import isfile, join, isdir
import time

PATH_DIRECTORY = os.path.abspath('../..')
PATH_TESTER_DIR = os.path.dirname('__file__')
PATH_SRC_DIR = '..'
sys.path.insert(0, os.path.abspath(os.path.join(PATH_TESTER_DIR, PATH_SRC_DIR)))

from tools.utils import load_image, show_image, print_prob_all, process_prob, metrics_multiclass
from tools.dataset import Dataset
from nets.vgg19 import cnn_vgg19

# tf.reset_default_graph()

PATH_GRAPH = PATH_DIRECTORY + '/graphs/'
PATH_TRAIN = PATH_DIRECTORY + '/data/example/leaf/train/'
PATH_TEST = PATH_DIRECTORY + '/data/example/leaf/test/'
npy_path_load_weigth = PATH_DIRECTORY + '/weights/vgg19/vgg19.npy'
npy_path_save_weigth = PATH_DIRECTORY + '/weights/vgg19/vgg19_train.npy'

npy_path_data = PATH_DIRECTORY + '/weights/data_fc6.npy'

# FUNCIONES
def test_model(net, session, objData, writer=None):
    label_total = []
    label_total_pred = []

    print('\n     # PHASE: Test classification')
    for i in range(objData.total_batchs_complete):
        batch, label = objData.generate_batch()
        # target = tf.one_hot(label, on_value=1, off_value=0, depth=net.num_class)
        # target = list(session.run(target))
        label_pred, acc, count, fc6 = session.run([net.labels_pred, net.accuracy, net.correct_count, net.fc6], feed_dict={vgg_batch: batch, vgg_label: label, train_mode: False})

        label_total = np.concatenate((label_total, label), axis=0)
        label_total_pred = np.concatenate((label_total_pred, label_pred), axis=0)

        total_batch = len(label)
        objData.next_batch_test()
        net.add_row_data_by_save(fc6, label, i)
        # writer.add_summary(summary, i)
        print('     results[ Total:'+str(total_batch)+' | True:'+str(count)+' | False:'+str(total_batch-count)+' | Accuracy:'+str(acc)+' ]')

    # Promediamos la presicion total
    print('\n     # STATUS:')
    y_true = label_total
    y_prob = label_total_pred
    accuracy_final = metrics_multiclass(y_true, y_prob)
    return accuracy_final

def train_model(net, session, objData, epoch, writer=None):
    print('\n     # PHASE: Training model')
    for ep in range(epoch):
        print('\n     Epoch:', ep)
        t0 = time.time()
        cost_i = 0
        for i in range(objData.total_batchs):
            batch, label = objData.generate_batch()

            # Run training
            t_start = time.time()
            _, cost = session.run([net.train, net.cost], feed_dict={vgg_batch: batch, vgg_label: label, train_mode: True})
            t_end = time.time()

            # Next slice batch
            objData.next_batch()
            # writer.add_summary(summary, i)

            cost_i = cost_i + cost
            print("     > Minibatch: %d train on batch time: %7.3f seg. - Loss: %7.4f" % (i, (t_end - t_start), cost))
            # print("     > Minibatch: %d train on batch time: %7.3f seg." % (i, (t_end - t_start)))
        
        t1 = time.time()
        print("     Cost per epoch: ", (cost_i / objData.total_batchs))
        print("     Time per iteration: %7.3f seg." % ((t1 - t0) / epoch))
        print("     Time epoch: %7.3f seg." % (t1 - t0))


# PARAMETERS

epoch = 3
mini_batch_train = 20
mini_batch_test = 5
learning_rate = 0.025
dim_last_layers = [4096, 4096, 4]

# GENERAMOS LOS DATOS 
data_train = Dataset(
    path_data=PATH_TRAIN + 'data_train.csv',
    path_dir_images=PATH_TRAIN,
    minibatch=mini_batch_train,
    cols=[0, 1],
    xtype='')

data_test = Dataset(
    path_data=PATH_TEST + 'data_test.csv',
    path_dir_images=PATH_TEST,
    minibatch=mini_batch_test,
    cols=[0, 1],
    random=False,
    xtype='')

with tf.Session() as sess:

    # VARIABLES
    vgg_batch = tf.placeholder(tf.float32, [None, 224, 224, 3])
    vgg_label = tf.placeholder(tf.int64, [None])
    train_mode = tf.placeholder(tf.bool)

    # MODEL VGG19
    vgg = cnn_vgg19(npy_path_load_weigth, learning_rate=learning_rate, load_weight_fc=False)
    vgg.build(vgg_batch, vgg_label, train_mode, dim_last_layers=dim_last_layers)
    sess.run(tf.global_variables_initializer())

    # EJECUTAMOS LA RED - TEST
    # test_writer = tf.summary.FileWriter(PATH_GRAPH + './vgg19/test', sess.graph)
    # train_writer = tf.summary.FileWriter(PATH_GRAPH + './vgg19/train', sess.graph)
    test_model(net=vgg, session=sess, objData=data_test)
    # vgg.save_data_npy(sess, npy_path_data)
    train_model(net=vgg, session=sess, objData=data_train, epoch=epoch)
    test_model(net=vgg, session=sess, objData=data_test)

    # SAVE WEIGHTs
    vgg.save_npy(sess, npy_path_save_weigth)
