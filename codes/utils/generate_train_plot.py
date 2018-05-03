#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import argparse
import matplotlib.pyplot as plt
import numpy as np
import cPickle as pickle
from os.path import dirname, basename, realpath
from scipy.interpolate import spline

def plot_train_subs(train_acc, val_acc, loss, title):
    
    dir_path = dirname(train_acc)+'/'
    train_acc = pickle.load(open(train_acc, 'r'))
    val_acc = pickle.load(open(val_acc, 'r'))
    loss = pickle.load(open(loss, 'r'))
    fig, axes = plt.subplots(nrows=2,ncols=1)
    #fig, axes = plt.subplots(nrows=2,ncols=1, figsize=(9,6))
    x = np.arange(1, len(train_acc)+1)
    location_plot1 = 'lower right'
    location_plot2 = 'upper right'

    print len(train_acc)

    axes[0].plot(x, train_acc, label='Train Accuracy', color='green')
    axes[0].plot(x,val_acc, label='Validation Accuracy', color='blue')
    axes[0].set_title(title)
    axes[0].grid(True)
    axes[0].legend(loc=location_plot1, ncol=2)
    axes[1].plot(x,loss,label='Train Loss', color='red')
    axes[1].legend(loc=location_plot2, ncol=1)
    axes[1].grid(True)    

    plt.setp(axes[0], xticks=(np.arange(1, max(x)+1)), yticks=(np.arange(0., 1.1, .2)))
    plt.setp(axes[1], xticks=(np.arange(1, max(x)+1)), yticks=(np.arange(0., max(loss)+.5, .5)))        

    for i in axes:
        ticklines = i.get_xticklines() + i.get_yticklines()
        gridlines = i.get_xgridlines() + i.get_ygridlines()
        ticklabels = i.get_xticklabels() + i.get_yticklabels()
        
        for line in ticklines:
            line.set_linewidth(2)

        for line in gridlines:
            line.set_linestyle('--')

        for label in ticklabels:
            label.set_color('black')
            label.set_fontsize('medium')
    
    axes[1].set_xlabel('Training Iterations (Epoch)')
    axes[1].set_ylabel('Loss')
    axes[0].set_ylabel('Accuracy (%)')

    plt.tight_layout()
    plt.grid(alpha=0.4)
    plt.savefig(dir_path+'plot_train_best_model.pdf')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('train_acc', metavar='train_acc_pickle',
                        help='pickle file containing the data related to the train accuracy during the train phase')
    parser.add_argument('val_acc', metavar='validation_acc_pickle',
                        help='pickle file containing the data related to the validation accuracy during the train phase')
    parser.add_argument('loss_train', metavar='loss_train_pickle',
                        help='pickle file containing the data related to the loss during the train phase')
    parser.add_argument('plot_title', metavar='title',
                        help='Title that you wish to have in the plot.')
    args = parser.parse_args()


    plot_train_subs(args.train_acc, args.val_acc, args.loss_train, args.plot_title)

