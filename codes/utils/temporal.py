#!/usr/bin/python
#-*- coding: utf-8 -*-

import argparse
import logging
import sys
from types import IntType
import numpy as np
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from os.path import join, realpath, dirname

import pathfile, progressbar


def window_size(nb_lines, nb_regions, fill_missing):
    """Calculates the number of elements in each window

    Parameters
    ----------
    nb_lines : int
        Number of images
    nb_regions : int
        Number of regions to divide the images
    fill_missing : boolean
        Fill the last window with copies of the last frame to keep the window alive

    Returns
    -------
    total : int
        total number of elements
    """
    assert type(nb_lines) is IntType and nb_lines > 0, \
           "number of lines is not an integer or <=0: %r" % nb_lines
    assert type(nb_regions) is IntType and nb_regions > 0, \
           "number of regions is not an integer or <=0: %r" % nb_regions
    nb_elements = nb_lines / nb_regions
    discrepancy = nb_lines % nb_regions
    nb_copies = 0
    if discrepancy != 0 and fill_missing:
        nb_copies = nb_regions - discrepancy
        total = nb_copies + nb_lines
    else:
        total = nb_lines - discrepancy
    return total


def generate_regions(nb_elements, nb_regions):
    """Create regions for all images
    
    Parameters
    ----------
    nb_elements : int
        Total number of elements to divide into regions
    nb_regions : int
        Number of regions

    Returns
    -------
    dreg : dict
        Dictionary containing the index of the image and its temporal region
    """
    assert type(nb_elements) is IntType and nb_elements > 0, \
           "number of elements is not an integer or <=0: %r" % nb_elements
    assert type(nb_regions) is IntType and nb_regions > 0, \
           "number of regions is not an integer or <=0: %r" % nb_regions

    dreg = {}
    ids = np.arange(1, nb_elements+1)
    mat = ids.reshape(nb_regions, nb_elements/nb_regions)
    rows, cols = mat.shape

    for r in range(rows):
        for c in range(cols):
            current = mat[r][c]
            if c == (cols-1):
                mv = mat[:,0]
            else:
                mv = mat[:,(c+1)]
            mv = list(mv)
            mv[r] = current
            dreg[current] = mv
    return dreg



def temporal_regions(input, output, nb_regions, fill_missing=False, join_mode='concat'):
    """From a file containing features from images, split them into temporal regions

    Parameters:
    -----------
    input : string
        File containing the path to an image and its features in each line
    output : string
        File to save features resulting of the temporal regions
    nb_regions : int
        Number of regions to divide the images
    fill_missing : boolean
        Fill the last window with copies of the last frame to keep the window alive
    join_mode : string (concat|mean)
        How to join features to output (concatenate features or calculate the mean)
    """
    input = realpath(input)
    output = realpath(output)
    assert type(nb_regions) is IntType and nb_regions > 0, \
           "number of regions is not an integer or <=0: %r" % nb_regions
    
    pf = pathfile.FileOfPaths(input)
    fout = open(output, 'w')

    # calculate the size of each window
    total_elements = window_size(pf.nb_lines, nb_regions, fill_missing)
    # get all sequences of images in regions 
    dregions = generate_regions(total_elements, nb_regions)

    pb = progressbar.ProgressBar(len(dregions.keys()))
    for i in sorted(dregions.keys()):
        line = ''
        id_line = i
        if i > pf.nb_lines:
            id_line = pf.nb_lines
        
        for c, nb_line in enumerate(dregions[i]):
            if nb_line > pf.nb_lines:
                nb_line = pf.nb_lines
            if join_mode == 'concat':
                # concatenation of the vectors of features
                if c == 0:
                    header = '%s %s' % pf.get_path(id_line, label=True)
                line += ' '+pf.features_line(nb_line, asstr=True)
            elif join_mode == 'mean':
                # calculate the mean of each cell of the vectors of features
                if c == 0:
                    header = '%s %s' % pf.get_path(id_line, label=True)
                    mean = map(float, pf.features_line(nb_line, asstr=False))
                    mean = np.array(mean)
                else:
                    feats = map(float, pf.features_line(nb_line, asstr=False))
                    feats = np.array(feats)
                    mean = (mean + feats)/2
                line = ' '.join(['%.4f' % elem for elem in mean])
            else:
                logger.error('Specify the mode of join (concat|mean)')
                sys.exit(1)
        fout.write('%s %s\n' % (header, line.strip()))
        pb.update()
    fout.close()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('featsfile', metavar='file_features', 
                        help='file containing features of images')
    parser.add_argument('nb_regions', metavar='number_regions', type=int, 
                        help='number of regions to split the images')
    parser.add_argument('-f', '--fill', action='store_true',
                        help='fill the last window with copies of the last frame')
    parser.add_argument('-m', '--mean', action='store_true',
                        help='generate the mean of the features in each temporal region')
    parser.add_argument('-o', '--output', default=None,
                        help='file to save the output features')
    args = parser.parse_args()
    
    if args.mean:
        join_mode = 'mean'
    else:
        join_mode = 'concat'

    input = realpath(args.featsfile)
    if args.output:
        output = args.output
    else:
        name = input.split('.')[0]
        fname = name+'_'+str(args.nb_regions)+'_'+join_mode+'.txt'
        output = join(dirname(input), fname)
    temporal_regions(input, output, args.nb_regions, args.fill, join_mode)
