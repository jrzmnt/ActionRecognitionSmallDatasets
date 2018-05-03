#!/usr/bin/python
#-*- coding: utf-8 -*-

import sys
sys.path.insert(0, '..')
import warnings
warnings.filterwarnings("ignore")
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


import argparse
from os.path import join, realpath, dirname, exists, basename, splitext
import numpy as np

from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import recurrent
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.normalization import BatchNormalization

from utils import pathfile

import temporal


class Model(object):
    def __init__(self, model, **kwargs):
        self.pars = self._defaultParameters()
        self.setParameters(**kwargs)
        self.model = model
        self.opt = self._optimizer(opt=self.pars['opt'])

    def _defaultParameters(self):
        """
        Return the default values for each parameter
        """
        return {
            'opt': 'sgd',
            'lr': 0.1,
            'decay': 1e-6,
            'loss': 'categorical_crossentropy',
            'batch_size': 64,
            'nb_epoch': 10,
            'shuffle': True,
            'momentum': 0.9,
            'nesterov': True,
            'rho': 0.95,
            'epsilon': 1e-08,
            'beta_1': 0.9,
            'beta_2': 0.999,
            'horizontal_flip': True,
            'im_size': 112,
            'nb_hidden': 2014,
            'input_dim': 1024,
            'input_length': 32,
            'nb_classes': 9,
            'dropout': 0.5
        }

    def setParameters(self, **kwargs):
        """
        Set parameters to the model
        """
        for key, val in kwargs.iteritems():
            self.pars[key] = val

    def _optimizer(self, opt):
        """
        Select the optimizer to the model.
        """
        optimizer = None
        opt = opt.lower()
        if opt == 'sgd':
            optimizer = SGD(lr=self.pars['lr'],
                            decay=self.pars['decay'],
                            momentum=self.pars['momentum'],
                            nesterov=self.pars['nesterov'])
        elif opt == 'rmsprop':
            optimizer = RMSprop(lr=self.pars['lr'],
                                rho=self.pars['rho'],
                                epsilon=self.pars['epsilon'])
        elif opt == 'adagrad':
            optimizer = Adagrad(lr=self.pars['lr'],
                                epsilon=self.pars['epsilon'],
                                decay=self.pars['decay'])
        elif opt == 'adadelta':
            optimizer = Adadelta(lr=self.pars['lr'],
                                 rho=self.pars['rho'],
                                 epsilon=self.pars['epsilon'],
                                 decay=self.pars['decay'])
        elif opt == 'adam':
            optimizer = Adam(lr=self.pars['lr'],
                             beta_1=self.pars['beta_1'],
                             beta_2=self.pars['beta_2'],
                             epsilon=self.pars['epsilon'],
                             decay=self.pars['decay'])
        return optimizer



# End of class Model

def LSTM(summary=False, nb_hidden=1024, nb_classes=10, input_dim=512, input_length=32, lr='1e-3', dropout=0.5):
    """
    Sepp Hochreiter Jurgen Schmidhuber (1997). "Long short-term memory".
    Neural Computation. 9 (8): 1735–1780. doi:10.1162/neco.1997.9.8.1735.
    """
    model = Sequential()

    model.add(recurrent.GRU(nb_hidden, return_sequences=True, input_dim=input_dim, input_length=input_length))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))


    model.add(recurrent.GRU(nb_hidden, return_sequences=True, input_dim=input_dim, input_length=input_length))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(recurrent.GRU(nb_hidden, return_sequences=True, input_dim=input_dim, input_length=input_length))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(recurrent.GRU(nb_hidden, return_sequences=True, input_dim=input_dim, input_length=input_length))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(recurrent.GRU(nb_hidden, return_sequences=True, input_dim=input_dim, input_length=input_length))
    model.add(BatchNormalization())
    model.add(Dropout(dropout))

    model.add(TimeDistributed(Dense(nb_classes)))
    model.add(Activation('softmax'))

    if summary:
        print(model.summary())
    return model


class LSTMModel(Model):
    """
    Class to load and run LSTM
    """

    def __init__(self, trainfile, valfile, **kwargs):
        """
        Initialize the class

        Parameters:
        -----------
        trainfile : string
            path to the file containing image, true label and features
        valfile : string
            path to the file containing image, true label and features
        """
        self.pars_mat = {'nb_frames': 32, 'stride': 16, 'save': True}
        self.pars_lstm = {'summary': False, 'nb_hidden': 512, 'nb_classes': 10,
                          'input_dim': 1024, 'input_length': 32, 'lr': 1e-3, 'dropout': 0.9}
        for arg in kwargs:
            if self.pars_mat.has_key(arg):
                self.pars_mat[arg] = kwargs[arg]
            if self.pars_lstm.has_key(arg):
                self.pars_lstm[arg] = kwargs[arg]

        self.dirin = dirname(trainfile)
        self.X_train, self.y_train = self._preprocess(trainfile, self.pars_mat['nb_frames'],
                                                      self.pars_mat['stride'], self.pars_mat['save'])

        self.X_val, self.y_val = self._preprocess(valfile, self.pars_mat['nb_frames'],
                                                  self.pars_mat['stride'], self.pars_mat['save'])

        batches, self.pars_lstm['input_length'], self.pars_lstm['input_dim'] = self.X_train.shape

        lstm = LSTM(**self.pars_lstm)
        Model.__init__(self, lstm, **kwargs)

    def _preprocess(self, fileinput, nb_frames, stride, save):
        """
        Generate `X` and `y` matrices
        """
        fileinput = realpath(fileinput)
        if fileinput.endswith('.npy'):
            X, y = pathfile.create_matrix_lstm(fileinput, frames=nb_frames, stride=stride, load_bin=True, save_in=None)
        else:
            if save:
                dirin = dirname(fileinput)
                fname, ext = splitext(basename(fileinput))
                save = join(dirin, fname + '.npy')
            X, y = pathfile.create_matrix_lstm(fileinput, frames=nb_frames, stride=stride, load_bin=False, save_in=save)
        newy = []
        for ydim in y:
            newy.append(np_utils.to_categorical(ydim, self.pars_lstm['nb_classes']))
        y = np.array(newy)
        return X, y

    def train(self):
        """
        From files containing the path to the images and their true label
        train and validate the model.

        Parameters:
        -----------
        trainfile : string
            path to the file containing images and true labels used in train mode
        valfile : string
            path to the file containing images and true labels used in validation mode
        """
        self.model.compile(loss=self.pars['loss'], optimizer=self.opt, metrics=['accuracy'])
        fcheckpoint = join(self.dirin, "lstm_weights.hdf5")
        checkpointer = ModelCheckpoint(filepath=fcheckpoint, monitor='val_acc', verbose=1, save_best_only=True,
                                       mode='auto')
        earlyStopping = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
        self.model.fit(self.X_train, self.y_train, batch_size=32, nb_epoch=self.pars['nb_epoch'],
                       validation_data=(self.X_val, self.y_val), callbacks=[checkpointer])#, earlyStopping])

    def test(self, filename):
        """
        From a file containing paths and true labels, predict the probabilities to each class
        """
        X_test, y_test = self._preprocess(filename, self.pars_mat['nb_frames'],
                                          self.pars_mat['stride'], self.pars_mat['save'])
        result = self.model.evaluate(X_test, y_test, verbose=1)
        print result
        #probs = self.model.predict_proba(X_test, batch_size=32, verbose=1)
        #np.save(join(self.dirin, 'probabitlies.npy'), probs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('filetrain', metavar='file_train', 
                        help='file containing training examples')
    parser.add_argument('filetest', metavar='file_test', 
                        help='file containing test examples')
    args = parser.parse_args()

    #recurrent_model(args.filetrain, args.filetest)
