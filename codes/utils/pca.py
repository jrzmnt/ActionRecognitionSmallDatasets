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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import pathfile
import progressbar


def discover_nb_components(X):
    """
    Check the variance to select the number of components to PCA
    
    Parameters
    ----------
    X : numpy.array
        matrix of features
    """
    u, s, v = np.linalg.svd(X.T)

    print X.shape
    print u.shape
    print s.shape
    print v.shape


def apply_pca(inputfile, fileout=None, **kwargs):
    """
    Using features from `inputfile`, reduces dimensionality using PCA.

    Parameters:
    -----------
    inputfile : string
        path to the file containing the frame, the true label and the list of features
    fileout : string
        path to the output file
    **kwargs : dict
        contains the parameters of the PCA to `sklearn`
        (n_components, copy, whiten, svd_solver, tol, iterated_power, random_state)
    """
    inputfile = realpath(inputfile)
    paths, y, X = pathfile.extract_vectors(inputfile, with_paths=True)
    X = StandardScaler().fit_transform(X)

    if not kwargs.has_key('n_components'):
        kwargs['n_components'] = min(X.shape)
        discover_nb_components(X)
    """
    if not fileout:
        dirout = dirname(inputfile)
        fname, ext = splitext(basename(inputfile))
        fileout = join(dirout, fname+'_'+str(kwargs['n_components'])+'.pca.txt')

    # reducing dimensionality
    logger.info('feeding PCA with data')
    X_pca = PCA(**kwargs)
    X_reduced = X_pca.fit_transform(X)
    logger.info('PCA fed with data')
    
    with open(fileout, 'w') as fout:
        for i, r in enumerate(X_reduced):
            vals = np.around(X_reduced[i], decimals=4)
            vals = map(str, vals)
            fout.write('%s %d %s\n' % (paths[i], y[i], ' '.join(vals)))   
    """

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('fileinput', metavar='input_file', 
                        help='file containing feature examples')
    #parser.add_argument('filetest', metavar='file_test', 
    #                    help='file containing test examples')
    args = parser.parse_args()

    apply_pca(args.fileinput)

