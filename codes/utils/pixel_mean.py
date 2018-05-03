#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This module calculates the RGB mean of a collection of images.
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger('image.rgbmean')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
import cv2
import numpy as np
from os.path import realpath, splitext, basename, join

from classes import progressbar, pathfile
import imresize

def getPixelMean(input, dirout, size=256, fname=None):
    """
    Generate a pixelwise mean for a file containing paths to images.

    Parameters:
    -----------
    input : string
        File containing the path to all images and their true labels
    dirout : string
        Path to the output folder

    Notes:
    -------
    The function generates three files:
        fname.binaryproto : contains the pixelwise mean of the images
        fname.npy : numpy array containing the mean
        fname.png : image resulting of the mean
    """
    from caffe.io import array_to_blobproto
    from skimage import io

    input = realpath(input)
    fnamein, ext = splitext(basename(input))
    dirout = realpath(dirout)
    if fname:
        fnameout = join(dirout, fname+'_mean')
    else:
        fnameout = join(dirout, fnamein+'_mean')
    mean = np.zeros((1, 3, size, size))

    
    
    logger.info('calculating mean for %d files.' % n)

    n = 251392    

    pb = progressbar.ProgressBar(n)
        
    
    with open(input) as imgs_path:

        for i in imgs_path:
            path = i
            img = io.imread(path)

            mean[0][0] += img[:, :, 0]
            mean[0][1] += img[:, :, 1]
            mean[0][2] += img[:, :, 2]
            pb.update()

    mean[0] /= n
    blob = array_to_blobproto(mean)
    with open(fnameout+'.binaryproto', 'wb') as f:
        f.write(blob.SerializeToString())
    np.save(fnameout+".npy", mean[0])
    meanImg = np.transpose(mean[0].astype(np.uint8), (1, 2, 0))
    io.imsave(fnameout+".png", meanImg)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='file_input', help='File containing paths to images and true labels')
    parser.add_argument('diroutput', metavar='dir_output', help='Folder to record files with the mean')
    parser.add_argument('size', metavar='size_img', help='Size of the images', type=int)
    args = parser.parse_args()

    getPixelMean(args.inputfile, args.diroutput, size=args.size)
