#!/usr/bin/python
#-*- coding: utf-8 -*-

"""
This script converts JPG files from a file containing paths to PPM files.
"""
import os
import logging
logger = logging.getLogger('convert_ppm')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
from PIL import Image
from os.path import isdir, join, dirname, basename, splitext

from progressbar import ProgressBar


def fix_path(path, folderout):
    #/usr/share/datasets/DogCentric/Play/Ball_Hime_5_4740_4875_frame_0.jpg
    old = '/usr/share/datasets/DogCentric/'
    fname, ext = splitext(basename(path))
    path = path.replace(old, folderout)
    path = join(dirname(path), fname+'.ppm')
    return path


def countlines(path):
    with open(path) as fin:
        for n, _ in enumerate(fin, start=1): pass
    return n


def convert_files(inputfile, outputfolder):
    """ Convert files """
    pb = ProgressBar(countlines(inputfile))
    fout = open(join(outputfolder, 'paths.txt'), 'w')
    with open(inputfile) as fin:
        for line in fin:
            path, label = line.strip().split()
            outpath = fix_path(path, outputfolder)
            if not isdir(dirname(outpath)):
                os.makedirs(dirname(outpath))
            img = Image.open(path)
            img.save(outpath, "PPM")
            fout.write('%s %s\n' % (outpath, label))
            pb.update()
    fout.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='file_input', 
                        help='path to the file containing images and true labels')  
    parser.add_argument('outputfolder', metavar='folder_output', 
                        help='path to the output folder')    
    args = parser.parse_args()
    convert_files(args.inputfile, args.outputfolder)
