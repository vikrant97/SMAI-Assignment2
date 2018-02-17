"""
Course: Statistical Methods in Artificial Intelligence (CSE471)
Semester: Fall '17
Professor: Gandhi, Vineet

Assignment 2: SVM using scikit-learn.
Skeleton code for implementing SVM classifier for a
character recognition dataset having precomputed features for
each character.

Dataset is taken from: https://archive.ics.uci.edu/ml/datasets/letter+recognition

Remember
--------
1) SVM algorithms are not scale invariant.
"""

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold.t_sne import TSNE

import argparse, os, sys

def get_input_data(filename):

    X = []; Y = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip().split(',')
            Y.append(line[0])
            X.append([float(x) for x in line[1:]])
    X = np.asarray(X); Y = np.asarray(Y)

    return X, Y

def SVM(train_data,
        train_labels,
        test_data,
        test_labels,
        kernel):

    model=svm.SVC(kernel=kernel,C=1,gamma=0.15,degree=2)
    model.fit(train_data,train_labels)
    
    test_predictions=model.predict(test_data)
    accuracy=accuracy_score(test_predictions,test_labels)
    precision=precision_score(test_predictions,test_labels,average='weighted')
    recall=recall_score(test_predictions,test_labels,average='weighted')
    f1=f1_score(test_predictions,test_labels,average='weighted')

    return accuracy, precision, recall, f1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default=None,
            help='path to the directory containing the dataset file')

    args = parser.parse_args()
    if args.data_dir is None:
        #print "Usage: python letter_classification_svm.py --data_dir='<dataset dir path>'"
        sys.exit()
    else:
        filename = os.path.join(args.data_dir, 'letter_classification_train.data')
        try:
            if os.path.exists(filename):
                pass
                #print "Using %s as the dataset file" % filename
        except:
            #print "%s not present in %s. Please enter the correct dataset directory" % (filename, args.data_dir)
            sys.exit()

    # Set the value for svm_kernel as required.
    svm_kernel = 'poly'

    X_data,Y_data=get_input_data(filename)

    sss = StratifiedShuffleSplit(n_splits=5, test_size=0.125)   # Do not change this split size
    accumulated_metrics = []
    fold = 1
    for train_indices, test_indices in sss.split(X_data, Y_data):
        # print "Fold%d -> Number of training samples: %d | Number of testing "\
        #     "samples: %d" % (fold, len(train_indices), len(test_indices))
        train_data, test_data = X_data[train_indices], X_data[test_indices]
        train_labels, test_labels = Y_data[train_indices], Y_data[test_indices]
        accumulated_metrics.append(
            SVM(train_data, train_labels, test_data, test_labels,
                svm_kernel))
        fold += 1

    accuracy=0
    precision=0
    recall=0
    f1=0 
    for row in accumulated_metrics:
        accuracy+=row[0]/5
        precision+=row[1]/5
        recall+=row[2]/5
        f1+=row[3]/5
    print "%0.7f,"%accuracy,"%0.7f,"%precision,"%0.7f,"%recall,"%0.7f"%f1

