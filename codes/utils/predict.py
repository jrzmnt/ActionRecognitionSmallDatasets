#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import argparse
import os
import tensorflow as tf
import numpy as np
import cPickle
from sklearn.metrics import *
import run
import architectures
import generator
from keras.models import load_model
import cv2
from keras.utils import np_utils
from os.path import dirname, basename, realpath
from keras.backend import manual_variable_initialization
from utils import pathfile, progressbar
from keras.preprocessing.image import *
from keras.applications.inception_v3 import InceptionV3, preprocess_input, decode_predictions
from keras.applications.vgg16 import VGG16
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.preprocessing import image
import sys

def _start(gpu):

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)


def _lets_do_it(model, test_file, w, mean_img):

    weight_name = basename(w)
    dir_out = dirname(w)+'/'
    #mean = cv2.imread("/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/mean.png")
    mean = cv2.imread(mean_img)
    pf = pathfile.FileOfPaths(test_file)
    pb = progressbar.ProgressBar(pf.nb_lines)
    dim_ordering = K.image_data_format()
    prediction = []
    pred_path = dir_out + 'prediction_' + weight_name + '.txt'
    fout = open(pred_path, 'w')

    with open(test_file) as f:
        for line in f:
            pb.update()
            path = line.split(' ')[0]
            y = line.split(' ')[1]


            img = image.load_img(path)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x -= mean
            preds = model.predict(x)
            class_predicted = np.argmax(preds)
            prediction.append([str(class_predicted),y])

            fout.write(path + ' ' + str(y) + ' ' + str(class_predicted) + '\n')
            
    return pred_path

def predict(network, weight, test, mean_img, nb_class):

    #starts the gpu 0    
    _start(1)

    if network == 'v3':
        base_model = InceptionV3(include_top=False)

    elif network == 'vgg16':
        base_model = VGG16(include_top=False)

    else:
        sys.exit(0)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.0)(x)
    predictions = Dense(int(nb_class), activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)

    opt = SGD(lr=0.0001, momentum=0.9, decay=1e-6, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    model.load_weights(weight)

    preds = _lets_do_it(model, test, weight, mean_img)

    #y = []
    #Y = []    
    #file = open(preds, 'r')
    #temp = file.read().splitlines()
    #for i in temp:
        #y.append(i[-3])
        #Y.append(i[-1])


    #print ''
    #print '-------------------------------------------------'
    #print classification_report(Y, y, digits=3)
    #print 'Accuracy:'+str(accuracy_score(Y,y))
    #print '-------------------------------------------------'



if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument('network', metavar='network',
                        help='Name of the network.')
    parser.add_argument('weight', metavar='weight',
                        help='file containing the weights to load the model.')
    parser.add_argument('images_path_file', metavar='images_path_file',
                        help='file containing the path to the images.')
    parser.add_argument('mean', metavar='mean',
                        help='Mean image of the dataset.')
    parser.add_argument('nb_class', metavar='nb_class',
                        help='Number of classes that will be predicted.')
    args = parser.parse_args()

    predict(args.network, args.weight, args.images_path_file, args.mean, args.nb_class)


