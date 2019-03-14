import matplotlib
matplotlib.use("TkAgg")
from skimage import data, io, segmentation, color, transform

try:
    from sklearn.metrics import confusion_matrix, average_precision_score, roc_curve, roc_auc_score, hamming_loss
    from sklearn.metrics import precision_recall_curve, f1_score, accuracy_score
    from sklearn.metrics import recall_score, precision_score, fbeta_score
except:
    pass

import numpy as np
from os import listdir
from os.path import isfile, join, isdir
import csv

PREPEND = '     '

def load_image(path, scale=255, xrange=[0, 1], dim_image=224):
    # load image
    img = io.imread(path)
    img = img / scale
    # assert (xrange[0] <= img).all() and (img <= xrange[1]).all(), "Los pixeles superan las cotas."
    # print "Original Image Shape: ", img.shape
    # we crop image from center
    short_edge = min(img.shape[:2])
    yy = int((img.shape[0] - short_edge) / 2)
    xx = int((img.shape[1] - short_edge) / 2)
    crop_img = img[yy: yy + short_edge, xx: xx + short_edge]
    # resize to 224, 224
    resized_img = transform.resize(crop_img, (dim_image, dim_image), mode='constant')
    return resized_img

def show_image (image):
    # print('show_image')
    io.imshow(image)
    io.show()

def load_data_npy (npy_path_data):
    data_dict = np.load(npy_path_data, encoding='latin1').item()
    data = np.concatenate([d[0] for (idx, d) in data_dict['data'].items()])
    labels = np.concatenate([d[1] for (idx, d) in data_dict['data'].items()])
    return data, labels

# returns the top1 string
def print_prob(prob, file_path):
    synset = [l.strip() for l in open(file_path).readlines()]
    # print prob
    pred = np.argsort(prob)[::-1]

    # Get top1 label
    print(PREPEND + '[RESULT]')
    top1 = synset[pred[0]]
    print(PREPEND + "#Top1: " + str(top1), prob[pred[0]])
    # Get top5 label
    top5 = [(synset[pred[i]], prob[pred[i]]) for i in range(5)]
    print(PREPEND + "#Top5: " + str(top5))
    return top1

def print_prob_all(prob, file_path, top=3):
    synset = [l.strip() for l in open(file_path).readlines()]
    print(PREPEND + '[RESULT]')
    for i in range(len(prob)):
        _prob = prob[i]
        pred = np.argsort(_prob)[::-1]
        top1 = synset[pred[0]]
        print(PREPEND + "#Top1: " + str(top1), _prob[pred[0]])

        if top > 0:
            topn = [(synset[pred[i]], _prob[pred[i]]) for i in range(top)]
            print(PREPEND + "#Top" + str(top) + ": ", topn)
        print('')


def generate_list_CSV(path_source, path_CSV_dest):

    with open(path_CSV_dest, mode='w') as data_file:
        data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        directories = listdir(path_source)
        _index = 0
        for (index, d) in enumerate(directories):
            subPath = join(path_source, d)
            if isdir(subPath):
                print(index, d)
                listFiles = listdir(subPath)
                for f in listFiles:
                    filename = join(subPath, f)
                    if isfile(filename) and filename.lower().endswith(('.jpeg', '.jpg', '.png')):
                        path_filename = join(d, f)
                        print(path_filename)
                        data_writer.writerow([path_filename, str(_index), d])
                _index = _index + 1


def process_prob(target, prob, predicted=[], plot_predicted=[]):

    total = len(target)
    count = 0
    num_class = len(prob[0])

    for i in range(total):
        true_result = np.argsort(prob[i])[::-1][0]
        if target[i] == true_result:
            count += 1

        predicted.append(true_result)
        plot_predicted.append(prob[i][true_result])

    accuracy = count / total
    print(PREPEND + 'results[ Total:'+str(total)+' | True:'+str(count)+' | False:'+str(total-count)+' | Accuracy:'+str(accuracy)+' ]')
    return count, predicted, plot_predicted

def metrics_multiclass(y_true=[], y_pred=[]):

    cm1 = confusion_matrix(y_true=y_true, y_pred=y_pred)
    total = sum(sum(cm1))

    accuracy = accuracy_score(y_true, y_pred)
    total_score = accuracy_score(y_true, y_pred, normalize=False)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    f1_weighted = f1_score(y_true, y_pred, average='weighted')

    print('Confusion Matrix : \n', cm1)  
    print('Total Correct: ', total_score, '/', total)
    print('Accuracy     : ', accuracy)
    print('F1-macro     : ', f1_macro)
    print('F1-micro     : ', f1_micro)
    print('F1-weighted  : ', f1_weighted)
    return accuracy