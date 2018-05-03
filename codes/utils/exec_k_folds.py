#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script divide paths in k-folds.
"""
import logging
logger = logging.getLogger('kfold.creating-kfolds')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import os
from os.path import join, realpath, exists, dirname, splitext, basename, isdir
import sys
import argparse


def create_dir(dir):
    """
    Create a directory if it's do not exist.
    :type dir: String
    :param dir: Path to the directory that will be created.
    :return: None.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)

def main(allpaths, k):

    dataset_file = realpath(allpaths)
    dir_dataset_file = dirname(allpaths)
    
    for fold in range(1,int(k)+1):
        
        logger.info('Preparing fold: %s' % fold)
        create_dir(dir_dataset_file+'/fold_'+str(fold))   
        
        train_file = open(dir_dataset_file+'/fold_'+str(fold)+'/train_file_fold_'+str(fold)+'.txt','w')
        test_file = open(dir_dataset_file+'/fold_'+str(fold)+'/test_file_fold_'+str(fold)+'.txt','w')

        with open(dataset_file, 'r') as dataset:
            
            for i in dataset:
                #print i
                #print i.split('/')[-2].split('_')[-2]
                act_fold = i.split('/')[-2].split('_')[-2]
                if int(act_fold) == fold:
                    #print act_fold, fold
                    test_file.write(i)
                else:
                    train_file.write(i)

        train_file.close()
        test_file.close()

    logger.info('Done!')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('allpath', metavar='allpath', 
                        help='file containing all the paths to the dataset')
    parser.add_argument('kfolds', metavar='numberOfKFolds', 
                        help='number of folds')
    args = parser.parse_args()
    main(args.allpath, args.kfolds)
