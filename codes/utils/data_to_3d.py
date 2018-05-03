import sys
sys.path.insert(0, '..')
import os
import argparse
import logging
from os.path import join, realpath, dirname, exists, basename, splitext
from Classes import pathfile, progressbar

logger = logging.getLogger('image.opticalflow')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


def write_in_file(file_to_write, path_list, y_list):

    for item in range(len(path_list)):
        file_to_write.write(path_list[item]+' '+y_list[item]+'\n')
    

def data_to_3d(inputfile, window):

    logger.info('Generating 3d data file from: %s' % inputfile)
    path_out = os.path.dirname(os.path.realpath(inputfile)) + '/' + os.path.splitext(inputfile)[0] + '_3d.txt'       
    WINDOW = int(window)
    TRESHOLD = int(WINDOW/4)
    treshold_counter = TRESHOLD
    img_counter = 0
    y_ = None
    path_list = []
    y_list = []
    fout = open(path_out, 'w')
    fin  = open(inputfile, 'r')
    pf = pathfile.FileOfPaths(inputfile)
    pb = progressbar.ProgressBar(pf.nb_lines)

    for line in fin:
        pb.update()
        path, y = line.strip().split()


        if y_ != y and treshold_counter == TRESHOLD:
            y_ = y
            path_list = []
            y_list = []
            path_list.append(path)
            y_list.append(y)
            treshold_counter = 0
            continue

        elif y_ != y and treshold_counter < TRESHOLD and len(path_list) >= len(path_list)-TRESHOLD:
            path_list.pop(0)
            y_list.pop(0)
            path_list.append(path_list[-1])
            y_list.append(y_list[-1])
            treshold_counter += 1
            write_in_file(fout, path_list, y_list)
            continue

        elif y_ == y and len(path_list) == WINDOW:
            path_list.pop(0)
            y_list.pop(0)
            path_list.append(path)
            y_list.append(y)
            write_in_file(fout, path_list, y_list)
            continue

        elif y_ == y and len(path_list) != WINDOW:
            path_list.append(path)
            y_list.append(y)

        if len(path_list) == WINDOW:
            write_in_file(fout, path_list, y_list)
            continue

    fout.close()
    fin.close()


if __name__ == "__main__":
    """
    description here
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='file_input', 
                        help='file containing paths, true labels and predicted labels')
    parser.add_argument('window', metavar='window', 
                        help='number of frames that will be send to a 3d model')
    args = parser.parse_args()
    data_to_3d(args.inputfile, args.window)
