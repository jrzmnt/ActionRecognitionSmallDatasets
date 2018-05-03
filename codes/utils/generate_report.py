#!/usr/bin/python
#-*- coding: utf-8 -*-
"""
This script runs svm using training and testing data
"""
import sys
sys.path.insert(0, '..')
import logging
logger = logging.getLogger('run.generate_report')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import argparse
import miscelaneous

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('inputfile', metavar='file_input', 
                        help='file containing predicted values')
    parser.add_argument('-o', '--fileout', metavar='file_output', 
                        help='path to the output file')
    args = parser.parse_args()
    miscelaneous.save_PRF(args.inputfile)
