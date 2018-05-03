#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
This module creates a file containing the path of the first image, path of the second image
and path to the output file as:

/home/user/Car/Car_Hime_2_6450_6590_frame_1.ppm /home/user/Car/Car_Hime_2_6450_6590_frame_3.ppm /home/user/OPF/Car/Car_Hime_2_6450_6590_frame_1.ppm

The generated file must be passed as argument to the C++ optical flow generator.
"""

import sys, os
sys.path.insert(0, '..')
import logging
logger = logging.getLogger('create_pathfile')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
from os.path import realpath, splitext, basename, join, exists, isdir, dirname


def paths_videos(path):
    """
    Create a dictionary containing the labels, paths and images from a text file.
    The dictionary has the structure of:
        dic[label] = {path_to_image: [image_names]}
    Thus, the examples below become:
        /usr/share/datasets/DogCentric/Sniff/Sniff_Ku_3_5630_5790_frame_1.jpg 0
        /usr/share/datasets/DogCentric/Sniff/Sniff_Ku_3_5630_5790_frame_10.jpg 0
        /usr/share/datasets/DogCentric/Sniff/Sniff_Ringo_3_3750_4020_frame_1.jpg 0
        /usr/share/datasets/DogCentric/Sniff/Sniff_Ringo_3_3750_4020_frame_15.jpg 0

        dic[0] = {'/usr/share/datasets/DogCentric/Sniff/Sniff_Ku_3_5630_5790':
                    ['_frame_1.jpg', '_frame_2.jpg, ... '_frame_10.jpg'],
                  '/usr/share/datasets/DogCentric/Sniff/Sniff_Ringo_3_3750_4020':
                    ['_frame_1.jpg', '_frame_2.jpg, ... '_frame_10.jpg']
        }
    """
    videos = {}
    n = 0
    with open(path) as fin:
        for n, line in enumerate(fin):
            path, label = line.strip().split()
            arr = path.split('_frame_')
            path_video = arr[0]
            fname = '_frame_'+arr[1]
            label = int(label)
            
            if videos.has_key(label):
                if videos[label].has_key(path_video):
                    videos[label][path_video].append(fname)
                else:
                    videos[label][path_video] = [fname]
            else:
                videos[label] = {path_video: [fname]}
    return videos, n


def get_window(dvideos, window):
    """
    Return pairs of images in order to generate the optical flow.
    These pairs contains the first and the last image of the optical flow
    according to the size of the window.
    """
    dout = {}
    for label in sorted(dvideos):
        for path in sorted(dvideos[label]):
            vimgs = dvideos[label][path]
            pairs = []
            for n, img in enumerate(vimgs):
                next_img = n + window
                if len(vimgs) > next_img:
                    pairs.append((img, vimgs[next_img]))
                elif n+1 != len(vimgs):
                    pairs.append((img, vimgs[-1]))
            if dout.has_key(label):
                dout[label][path] = pairs
            else:
                dout[label] = {path: pairs}
    return dout

    
def fix_path(path, outputfolder):
    """
    Change the path of the file to save in output
    """
    folder_path = '/'.join(path.split('/')[:-2])
    path = path.replace(folder_path, dirname(outputfolder))
    return path


def create_file(inputfile, outputfolder, window=2):
    """
    Create a file containing the path of the input images and the output
    image.
    """
    logger.info('Generating optical flow from: %s' % inputfile)
    inputfile = realpath(inputfile)
    dirout = dirname(outputfolder)
    
    fout = open(join(dirout, 'paths_cpp.txt'), 'w')
    dvideos, size = paths_videos(inputfile)
    dpairs = get_window(dvideos, window)

    for label in sorted(dpairs):
        for path in sorted(dpairs[label]):
            for img1, img2 in sorted(dpairs[label][path]):
                frame1 = '%s%s' % (path, img1)
                frame2 = '%s%s' % (path, img2)
                frameout = fix_path(frame1, outputfolder)
                print frameout
                if not isdir(dirname(frameout)):
                    os.makedirs(dirname(frameout))
                fout.write('%s %s %s\n' % (frame1, frame2, frameout))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='file_input', 
                        help='file containing paths, true labels and predicted labels')
    parser.add_argument('outputfolder', metavar='folder_output', 
                        help='path to the output folder')
    args = parser.parse_args()
    create_file(args.inputfile, args.outputfolder)
