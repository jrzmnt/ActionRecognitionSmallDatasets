#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import os
import argparse
from os.path import join, realpath, dirname, exists, basename, splitext
from sklearn import svm
import numpy as np

import pathfile
import progressbar


def classifier_svm(ftrain, ftest, fileout=None, **kwargs):
    """
    Using features from `ftrain`, train a SVM classifier and uses `ftest` to 
    predict its labels.

    Parameters:
    -----------
    ftrain : string
        path to the file containing the frame, the true label and the list of features
    ftest : string
        path to the file containing the frame, the true label and the list of features
    fileout : string
        path to the output file
    **kwargs : dict
        contains the parameters of the SVM classifier to SKlearn
        (C, cache_size, class_weight, coef0, decision_function_shape, degree, gamma, 
         kernel, max_iter, probability, random_state, shrinking, tol, verbose)
    """
    ftrain = realpath(ftrain)
    ftest = realpath(ftest)

    if not fileout:
        dirout = dirname(ftrain)
        fname, ext = splitext(basename(ftrain))
        fileout = join(dirout, fname+'.svm.txt')

    # training phase
    _, y, X = pathfile.extract_vectors(ftrain, with_paths=False)
    logger.info('feeding SVM with data in training phase')
    clf = svm.SVC(**kwargs)
    clf.fit(X, y)

    # testing phase
    paths, y_test, X_test = pathfile.extract_vectors(ftest, with_paths=True)
    logger.info('predicting classes for testing vector')
    pred = clf.predict(X_test)

    logger.info('saving output file as: %s' % fileout)
    with open(fileout, 'w') as fout:
        for path, label, p in zip(paths, y_test, pred):
            fout.write('%s %d %d\n' % (path, label, p))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filetrain', metavar='file_train', 
                        help='file containing training examples')
    parser.add_argument('filetest', metavar='file_test', 
                        help='file containing test examples')
    args = parser.parse_args()

    classifier_svm(args.filetrain, args.filetest)
