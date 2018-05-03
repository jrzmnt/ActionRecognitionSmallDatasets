#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import argparse
import numpy as np
from keras.models import Model
from keras.preprocessing import image
import architectures
from os.path import dirname, basename
import pathfile, progressbar
import cv2


def _start(gpu):
    import os
    import tensorflow as tf
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]=str(gpu)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    sess = tf.Session(config=config)
    
    return 'Done!'


def extract(net, weight, imgs_path, mean_path, classes):

    if net == 'vgg16' or net == 'inception_v3':
        #layer_name = 'dense_1'
        #layer_name = 'global_average_pooling2d_1'
        layer_name = 'dense_2'

    elif net == 'smallnet':
        sys.exit(0)

    elif net == 'c3d':
        sys.exit(0)

    else:
        raise ValueError('Network "%s" not found!' % (net))

    m, params = architectures.get_model(net, nb_classes=int(classes))
    m.load_weights(weight)

    intermediate_layer_model = Model(inputs=m.input, outputs=m.get_layer(layer_name).output)

    weight_name = basename(weight)
    data_name = basename(imgs_path)
    dir_out = dirname(weight) + '/'
    name_out = dir_out+net+'_'+data_name+'_'+'extracted_features_from_'+weight_name+'.txt'
    file_out = open(name_out,'w')

    pf = pathfile.FileOfPaths(imgs_path)
    pb = progressbar.ProgressBar(pf.nb_lines)

    np.set_printoptions(precision=4)
    mean = cv2.imread(mean_path)

    with open(imgs_path) as f:
        _start(0)

        for line in f:
            pb.update()
            img_path, y = line.split()
            img = image.load_img(img_path)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x -= mean
            intermediate_output = intermediate_layer_model.predict(x)
            file_out.write(imgs_path + ' ' + y)

            for i in np.nditer(intermediate_output):
                file_out.write(' '+str(np.round(i,4)))

            file_out.write('\n')

    file_out.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('network', metavar='network',
                        help='string containing the name of the network, e.g.:vgg16, xception.')
    parser.add_argument('weight', metavar='weight',
                        help='file containing the weights to load the model.')
    parser.add_argument('images_path_file', metavar='images_path_file',
                        help='file containing the path to the images.')
    parser.add_argument('mean_path', metavar='mean_path',
                        help='file containing the mean of the images.')
    parser.add_argument('classes', metavar='classes',
                        help='number of the classes of the used dataset')


    args = parser.parse_args()

    extract(args.network, args.weight, args.images_path_file, args.mean_path, args.classes)


