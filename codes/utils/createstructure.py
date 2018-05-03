#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script fixes all paths and true labels for all activity files.
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger('structure.create-structure')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

import argparse
from os.path import join, realpath


DATASET = ['basketball',
           'biking',
           'diving',
           'golf_swing',
           'horse_riding',
           'soccer_juggling',
           'swing',
           'tennis_swing',
           'trampoline_jumping',
           'volleyball_spiking']

FOLDERS = ['01', '02', '03', '04', '05',
           '06', '07', '08', '09', '10',
           '11', '12', '13', '14', '15',
           '16', '17', '18', '19', '20',
           '21', '22', '23', '24', '25']

def main(inputfolder):
    """
    Open all files and create the real path with the true label for each image
    """
    inputfolder = realpath(inputfolder)
    for data in DATASET:
        for fol in FOLDERS:
            actfile = join(inputfolder, data, data+'.txt')
            logger.info('Changing data in: %s' % actfile)
            filedata = []
            with open(actfile) as fin:
                for line in fin:
                    id, y = map(int, line.strip().split('\t'))
                    if y == -1000:
                        y = 0
                    path = join(inputfolder, 'data'+str(data), action, 'original', str(id)+'.jpg')
                    filedata.append((path, y))
                path = join(inputfolder, 'data'+str(data), action, 'original', str(id+1)+'.jpg')
                filedata.append((path, y))
            with open(actfile, 'w') as fout:
                for path, y in filedata:
                    fout.write('%s %d\n' % (path, y)) 
                       

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfolder', metavar='infolder', 
                        help='path to the root folder where all files where extracted')
    args = parser.parse_args()
    main(args.inputfolder)
