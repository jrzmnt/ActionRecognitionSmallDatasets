#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This module generates images with optical flow from a collection of images.
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger('image.opticalflow')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
import cv2
import numpy as np
import progressbar
import pathfile

from skimage import io
from os.path import realpath, splitext, basename, join, exists




def optical_flow(frame1, frame2):
    """
    Generates the optical flow between two images.
    
    Parameters:
    -----------
    frame1 : string
        path to the first image
    frame2 : string
        path to the second image
    """
    print frame1
    print frame2
    if not (exists(frame1) and exists(frame1)):
        logger.error('cannot find images: %s %s' % (frame1, frame2))
        sys.exit(0)
    img1 = cv2.imread(frame1)
    img2 = cv2.imread(frame2)
    gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    hsv = np.zeros_like(img1)
    hsv[...,1] = 255

    #flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, flow=None, pyr_scale=0.5, 
                                        levels=3, winsize=10, iterations=5, 
                                        poly_n=5, poly_sigma=1.5, flags=0)
    mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
    hsv[...,0] = ang*180/np.pi/2
    hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    bgr = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
    return bgr


def window_optical_flow(vec, window):
    """
    Return pairs of images to generate the optical flow.
    These pairs contains the first and the last image of the optical flow
    according to the size of the window.
    
    Parameters:
    -----------
    vec : array_like
        sorted list containing the image ids (type int)
    window : int
        size of the window to generate the optical flow

    Returns:
    --------
    pairs : array_like
        list containing tuples with pairs as (first image, last image)
        of the optical flow

    Usage:
    ------
    >>> vec = [0, 1, 2, 3]
    >>> window_optical_flow(vec, 2)
        [(0, 2), (1, 3), (2, 3), (3, 3)]
    """
    pairs = []
    for img in vec:
        last_img = img + window
        if last_img in vec:
            pairs.append((img, last_img))
        else:
            pairs.append((img, vec[-1]))
    return pairs


def opticalflow_from_file(inputfile, window=2):
    """
    Open the inputfile and generates the optical flow for all images in the file

    Parameters:
    -----------
    inputfile : string
        path to the file containing the image paths
    """
    logger.info('Generating optical flow from: %s' % inputfile)
    inputfile = realpath(inputfile)
    pf = pathfile.PathFile(inputfile, imlist=True)
    pb = progressbar.ProgressBar(pf.numberFiles())
    for data, activity, imsize, img, labels in pf:
        pairs = window_optical_flow(img, window)
        for img1, img2 in pairs:
            frame1 = join(pf.localPath(), 'data'+str(data), activity, imsize, str(img1)+'.jpg')
            frame2 = join(pf.localPath(), 'data'+str(data), activity, imsize, str(img2)+'.jpg')
            imgof = optical_flow(frame1, frame2)
            
            pathout = join(pf.localPath(), 'data'+str(data), activity, 'ofl256', str(img1)+'.jpg')
            pathfile.createFoldersFromPath(pathout)
            cv2.imwrite(pathout, bgr)
            pb.update()
    cv2.destroyAllWindows()
    logger.info('Created %d files with optical flow' % pf.numberFiles())


if __name__ == "__main__":
    """
    sss
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='file_input', 
                        help='file containing paths, true labels and predicted labels')
    args = parser.parse_args()
    opticalflow_from_file(args.inputfile)
