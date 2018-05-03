# coding=utf-8

import sys
import keras
import numpy as np
import generator
import cPickle
import os
import cv2
import logging
from keras.utils import np_utils
from os.path import dirname, basename, realpath
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
from utils import pathfile, progressbar
from keras.preprocessing.image import *
from keras import backend as K
logger = logging.getLogger('models.keras.preprocessing')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
handler = logging.FileHandler('debug.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
sys.path.insert(0, '..')

def get_path(network):
    """
    Select the right path given the chosen network.
    :type network: String
    :param network: Name of the desired network, e.g.: 'vgg16','xception'.
    :return: the path to the Experiments folder given the chosen network.
    """

    # Just verify wich network was passed to the function and address the right path
    if network == 'vgg16':
        path = 'experiments/vgg-16/'

    elif network == 'smallnet':
        path = 'experiments/smallnet/'

    elif network == 'inception_v3':
        path = 'experiments/inception_v3/'

    elif network == 'c3d':
        path = 'experiments/c3d/'

    elif network == 'dense':
        path = 'experiments/dense/'

    elif network == 'lstm':
        path = 'experiments/lstm/'

    else:
        raise ValueError('Network "%s" not found!' % (network))

    return path


def create_dir(dir):
    """
    Create a directory if it's do not exist.
    :type dir: String
    :param dir: Path to the directory that will be created.
    :return: None.
    """
    if not os.path.exists(dir):
        os.makedirs(dir)


def _lstm_preprocess(fileinput, params):
    """
    Prepare the input for lstm
    :type model: keras.model
    :param model: Selected network.
    :param params: Dict
    :param params: Dictionary of parameters used in the model.
    :return: None
    """
    fileinput = realpath(fileinput)

    X, y = pathfile.create_matrix_lstm(fileinput,
                                       frames=params['nb_frames'],
                                       stride=params['stride'])

    newy = []

    for ydim in y:
        newy.append(np_utils.to_categorical(ydim, params['nb_classes']))

    y = np.array(newy)
    return X, y

def train(model, network, train_file, validation_file, params):
    """
    Prepare and Train the model.
    :type model: keras.model
    :param model: Selected network.
    :type network: String
    :param network: Name of the desired network, e.g.: 'vgg16','xception'.
    :type train_file: String
    :param train_file: Path to the file containing the image paths and their classes.
    :type validation_file: String
    :param validation_file: Path to the file containing the image paths and their classes.
    :param params: Dict
    :param params: Dictionary of parameters used in the model.
    :return: None
    """
    logger.info('Start the train with Net:%s, Drop:%.2f, Lr:%.6f' % (network, params['dropout'], params['lr']))

    # Setting the directories that the weights will be saved
    path = get_path(str(network))
    drop = params['dropout']
    drop_directory = str(path)+str(drop)
    lr_directory = str(drop_directory+'/'+str(params['lr']))

    # Call the function to create the directories
    create_dir(drop_directory)
    create_dir(lr_directory)


    if params['temporal']:
        # Create the generators
        train_gen_flow = generator.CreateTemporalGen(train_file, params)
        validation_gen_flow = generator.CreateTemporalGen(validation_file, params)

    elif params['lstm']:
        # Adjust features for lstm training
        X_train, y_train = _lstm_preprocess(train_file,params)
        X_val, y_val = _lstm_preprocess(validation_file, params)
        batches, params['input_size'], params['input_dim'] = X_train.shape

    elif params['volume']:
        train_gen_flow = generator.CreateGen3D(train_file, params)
        validation_gen_flow = generator.CreateGen3D(validation_file, params)

    else:
        # Create the generators
        train_gen = generator.create_gen()
        validation_gen = generator.create_gen()
        train_gen_flow = train_gen.flow_from_file(train_file, shuffle=True)
        validation_gen_flow = validation_gen.flow_from_file(validation_file, shuffle=False)

    # Create the Monitors
    #file_path = lr_directory+"/weights.{epoch:02d}-{val_loss:.2f}.hdf5"
    file_path = lr_directory+"/weights.hdf5"
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max', save_weights_only=True)
    acc_loss_monitor = History()
    earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='min')

   

    if params['lstm'] is False:
        # Counts how many images train and validation have
        nb_imgs_train = train_gen_flow.nb_sample
        nb_imgs_validation = validation_gen_flow.nb_sample

        #from keras.backend import manual_variable_initialization
        #manual_variable_initialization(True)
        # Call the fit_generator passing the necessary params
        model.fit_generator(train_gen_flow,
                            verbose=1,
                            steps_per_epoch=nb_imgs_train / params['batch_size'],
                            epochs=params['nb_epoch'],
                            validation_data= validation_gen_flow,
                            validation_steps=nb_imgs_validation / params['batch_size'],
                            callbacks=[checkpoint, acc_loss_monitor, earlyStopping],
                            workers=0
                            )

    else:
        #model.compile(loss=params['loss'], optimizer=params['opt'], metrics=['acc'])



        model.fit(X_train,
                  y_train,
                  batch_size=params['batch_size'],
                  nb_epoch=params['nb_epoch'],
                  validation_data=(X_val, y_val),
                  callbacks=[checkpoint, acc_loss_monitor, earlyStopping])

    # Saving the train and validation accuracy, and the loss from the training.
    acc_hist = acc_loss_monitor.history['acc']
    val_acc_hist = acc_loss_monitor.history['val_acc']
    loss_hist = acc_loss_monitor.history['loss']

    acc_pkl_path = lr_directory + '/acc_hist_drop' + str(params['dropout']) + '_lr' + str(params['lr']) + '.p'
    loss_pkl_path = lr_directory + '/loss_hist_drop' + str(params['dropout']) + '_lr' + str(params['lr']) + '.p'
    val_acc_pkl_path = lr_directory + '/val_acc_hist_drop' + str(params['dropout']) + '_lr' + str(params['lr']) + '.p'

    cPickle.dump(acc_hist, open(acc_pkl_path, 'wb'))
    cPickle.dump(val_acc_hist, open(val_acc_pkl_path, 'wb'))
    cPickle.dump(loss_hist, open(loss_pkl_path, 'wb'))

    # plot_name = 'drop_'+str(params['dropout']) + '_lr' + str(params['lr'])+'_train'
    # plotTrain.plot(acc_pkl_path, loss_pkl_path, val_acc_pkl_path, plot_name)


def test(model, network, test_file, params, weights=None):
    """
    Test the model and obtain the resulting accuracy.
    :type model: keras.model
    :param model: Selected network.
    :type weights: hdf5
    :param weights: Weights to be loaded in the model.
    :type network: String
    :param network: Network name to save it in the correct folder.
    :type test_file: String
    :param test_file: Path to the file containing the image paths and their classes.
    :param params: Dict
    :param params: Dictionary of parameters used in the model. 
    :return: None
    """

    if params['lstm']:
        X_test, y_test = _lstm_preprocess(test_file, params)
        print model.predict(X_test)

        #values = model.evaluate(X_test, y_test, verbose=1)

    else:
        # Create generator for test.
        test_gen = generator.create_gen()
        test_gen_flow = test_gen.flow_from_file(test_file, shuffle=False)

        # Set parameters to evaluate.
        nb_imgs_test = test_gen_flow.nb_sample

        # Evaluate using test_gen.
        #model.load_weights(weights)
        values = model.evaluate_generator(test_gen_flow, nb_imgs_test / params['batch_size'], workers=0)

    return values


def predict(model, test_file, weights, params):
    """
    Predicts images from a file (test_file) and save the prediction in a .txt file.
    :type model: keras.model
    :param model: Selected network.
    :type test_file: String
    :param test_file: Path to the file containing the image paths and their classes.
    :type weights: hdf5
    :param weights: Weights to be loaded in the model.
    :return: None
    """
    # Extract the name and the folder from the given weights file
    weight_name = basename(weights)
    dir_out = dirname(weights)+'/'
    mean = cv2.imread("/home/juarez/ActionRecognitionSmallDatasets/codes/Data/dogcentric/dog_centric_mean.png")
    # Create a progress bar in order to controle how long predict is going to take
    pf = pathfile.FileOfPaths(test_file)
    pb = progressbar.ProgressBar(pf.nb_lines)
    dim_ordering = K.image_data_format()
    # Create a txt to be written with the prediction
    fout = open(dir_out + 'prediction_' + weight_name + '.txt', 'w')

    #X_test, y_test = _lstm_preprocess(test_file, params)
    model.load_weights(weights)
    # Read each image from test_file and predict it with the model
    with open(test_file) as f:
        for line in f:
            pb.update()
            path = line.split(' ')[0]
            y = line.split(' ')[1][0]
            img = cv2.imread(path)
            img = np.reshape(img, [1, 240, 240, 3])
            #img = np.reshape(img, dim_ordering)
            #img = img_to_array(img)
            #img -= mean
            classes = model.predict(img)
            prediction = classes.argmax()
            #print str(prediction) + ' - ' + str(y)
            fout.write(path + ' ' + str(y) + ' ' + str(prediction) + '\n')
    fout.close()
