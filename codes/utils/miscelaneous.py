#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
This file contains miscelaneous functions that are accessed by classes and other functions.
"""

import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger('misc.miscelaneous')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from os.path import join, realpath, dirname, exists, basename, splitext
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np
import pathfile, progressbar

def filter_softmax_topN(inputfile, fileout, N=9, max_diff=0.1):
    """
    Read the input file and keeps only the label with the highest score.
    In case of the difference between scores be less than ``difference'', 
    verify whether one of the top labels is equal to the last label. In case
    true, assign the last label to the target label. In case false, keeps 
    the label of the top1 as the selected label.

    Parameters:
    -----------
    inputfile : string
        path to the file containing the frame, the true label and the 
        predicted label
    fileout : string
        path to the output file
    N : int
        number of elements in softmax output
    max_diff : float
        maximum difference between two labels to verify the target label
    """
    inputfile = realpath(inputfile)
    fileout = realpath(fileout)
    fout = open(fileout, 'w')

    logger.info('loading text file: %s' % inputfile)
    pf = pathfile.PathFile(inputfile, imlist=False)
    last_label = 0
    for _ in pf:
        path = pf.path
        y = pf.label
        top1, top2 = pf.current_features(N=2)
        c1, val1 = top1
        c2, val2 = top2

        difference = val1 - val2
        if difference > max_diff:
            fout.write('%s %d %d\n' % (path, y, c1))
            last_label = c1
        else:
            if last_label == c2:
                fout.write('%s %d %d\n' % (path, y, c2))
                last_label = c2
            else:
                fout.write('%s %d %d\n' % (path, y, c1))
                last_label = c1


ACTIV_KSCGR = ['None', 'Breaking', 'Mixing', 'Baking', 'Turning', 'Cutting', 'Boiling', 'Seasoning', 'Peeling']
ACTIV_DOG = ['Car', 'Drink', 'Feed', 'Look left', 'Look right', 'Pet', 'Play with ball', 'Shake', 'Sniff', 'Walk']
ACTIV_UCF = ['0','1','2','3','4','5','6','7','8','9','10','11']

def accuracy_per_class(cm):
    row, col = cm.shape
    lineVal, colVal, accuracy = [], [], []
    colVal.append(cm.sum(axis=0, dtype='float'))
    lineVal.append(cm.sum(axis=1, dtype='float'))

    for i in range(row):
        accuracy.append(cm[i][i] / (colVal[0][i] + lineVal[0][i] - cm[i][i]))

    print 'Acc. per class:'+ str(accuracy)
    acc_per_class_norm = np.round_(accuracy, decimals=3)
    return acc_per_class_norm


def save_PRF(inputfile, outputfile=None):
    """
    Create a report containing Precision, Recall and F-measure
    of all classes
    """
    inputfile = realpath(inputfile)
    dirin = dirname(inputfile)
    fname, ext = splitext(inputfile)

    if not outputfile:
        outputfile = join(dirin, fname+'_report.txt')
    fout = open(outputfile, 'w')

    ally,allp = [], []
    with open(inputfile) as fin:
        for line in fin:
            _, y, pred = line.strip().split()
            ally.append(int(y))
            allp.append(int(pred))

    if len(set(ally)) == 9:
        labels = ACTIV_KSCGR
    elif len(set(ally)) == 10:
        labels = ACTIV_DOG
    elif len(set(ally)) == 11:
        labels = ACTIV_UCF
    else:
        labels = ACTIV_KSCGR

    ACTIV = [labels[cl] for cl in set(ally).union(allp)]

    fout.write(classification_report(ally, allp, target_names=ACTIV))
    fout.write('\n\n')
    acc = accuracy_score(ally, allp)
    cm = confusion_matrix(np.array(ally, dtype=float), np.array(allp, dtype=float))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_round = np.round_(cm_normalized, decimals=3)
    acc_norm = accuracy_per_class(cm)
    fout.write(np.array2string(cm))
    fout.write('\n\n')
    fout.write(np.array2string(cm_round))
    fout.write('\n\n')
    fout.write('Accuracy: '+str(acc))
    fout.write('\n\n')
    fout.write('Accuracy Per Class: ')
    fout.write(np.array2string(acc_norm))


def vectors_training(inputfile):
    """
    Generates the training vectors from the input file

    Parameters:
    -----------
    inputfile : string
        path to the file containing the frame, the true label and the list of softmax
    """
    X, y = [], []
    pf = pathfile.FileOfPaths(inputfile)
    pb = progressbar.ProgressBar(pf.nb_lines)
    for _ in pf:
        X.append(pf.feats)
        y.append(pf.label)
        pb.update()
    X = np.array(X).astype(float)
    y = np.array(y).astype(int)
    return X, y


def save_line(fout, path, y, array):
    """
    Save a line into `fout`. The line contains
        path y array
    
    Parameters:
    -----------
    fout : string
        instance of the output file
    path : string
        path to the image
    y : int
        true label
    array : array_like
        list containing features, softmax or other values
    """
    line = ''
    for el in array:
        line += str(el)+' '
    fout.write('%s %d %s\n' % (path, y, line[:-1]))
    return fout


def joinVectorsInFile(file1, file2, outfile):
    """
    Given two files with path, true label and features, join
    both files into one file containing path, true label and full features
    
    Parameters:
    -----------
    file1 : string
        path to the input file
    file2 : string
        path to the input file
    outfile : string
        path to the output file
    """
    fout = open(outfile, 'w')
    pf = pathfile.FileOfPaths(file1)
    logger.info('creating file: %s' % outfile)
    pb = progressbar.ProgressBar(pf.nb_lines)
    with open(file2) as fin2:
        for line1, line2 in zip(pf, fin2):
            feats1 = map(str, pf.feats)
            feats1 = ' '.join(feats1)
            fout.write('%s %s\n' % (line2.strip(), feats1))
            pb.update()
    pf.close()
    fout.close()

