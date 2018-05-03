#!/usr/bin/python
# coding=utf-8

from keras.models import Model, Sequential
from keras.layers import Input, recurrent
from keras.layers.wrappers import TimeDistributed
from keras.layers.pooling import GlobalAveragePooling2D
from keras.layers.core import Dense, Dropout, Flatten, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.layers import merge
from customlayers import crosschannelnormalization, splittensor
from keras.layers.convolutional import ZeroPadding2D
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam

def _default_parameters():
    """
    Return the default values for each parameter
    """

    return {
        'opt': 'adadelta',
        'activation_function': 'softmax',
        'lr': 0.0001,
        'decay': 1e-6,
        'loss': 'categorical_crossentropy',
        'batch_size': 32,
        'nb_epoch': 20,
        'shuffle': True,
        'momentum': 0.9,
        'nesterov': True,
        'rho': 0.95,
        'epsilon': 1e-08,
        'beta_1': 0.9,
        'beta_2': 0.999,
        'horizontal_flip': False,
        'im_size': 240,#256,
        'dense_layer': 1024,
        'nb_classes': 10,
        'nb_channels': 3,
        'dropout': 0.5,
        'metrics': ['accuracy'],
        'volume': None,
        'input_size': 25,
        'temporal': False,
        'input_dim': 512,
        'nb_frames': 60,
        'stride': 16,
        'nb_hidden':512,
        'lstm': False

    }

def get_model(network, weights=None, pre_trained=False, **kwargs):
    params = _default_parameters()

    if kwargs:
        # Set params.
        for key, val in kwargs.iteritems():
            params[key] = val

    net = network.lower()

    if net == 'vgg16':
        model = get_vgg16(pre_trained, params, weights)

    elif net == 'alexnet':
        model = get_alex(params, weights)

    elif net == 'smallnet':
        model = get_small(params, weights)

    elif net == 'dense':
        model = get_dense(params, weights)

    elif net == 'inception_v3':
        model = get_inception(pre_trained, params, weights)

    elif net == 'c3d':
        model = get_c3d(pre_trained, params, weights)

    elif net == 'lstm':
        model = get_lstm(params, weights)

    else:
        raise ValueError('Network "%s" not found!' % (net))

    # Set the optimizer.
    opt_name = params['opt'].lower()
    if opt_name == 'sgd':
        opt = SGD(lr=params['lr'], momentum=params['momentum'], decay=params['decay'], nesterov=params['nesterov'])
    elif opt_name == 'rmsprop':
        opt = RMSprop(lr=params['lr'], rho=params['rho'], epsilon=params['epsilon'], decay=params['decay'])
    elif opt_name == 'adagrad':
        opt = Adagrad(lr=params['lr'], epsilon=params['epsilon'], decay=params['decay'])
    elif opt_name == 'adadelta':
        opt = Adadelta(lr=params['lr'], rho=params['rho'], epsilon=params['epsilon'], decay=params['decay'])
    elif opt_name == 'adam':
        opt = Adam(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'], epsilon=params['epsilon'], decay=params['decay'])
    elif opt_name == 'adamax':
        opt = Adamax(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'], epsilon=params['epsilon'], decay=params['decay'])
    elif opt_name == 'nadam':
        opt = Nadam(lr=params['lr'], beta_1=params['beta_1'], beta_2=params['beta_2'], epsilon=params['epsilon'])
    else:
        raise ValueError('Optimizer "{}" not found.'.format(opt_name))


    model.compile(loss=params['loss'],
                  optimizer=opt,
                  metrics=params['metrics'])

    return model, params


def get_inception(pre_trained, params, weights):
    if pre_trained:
        base_model = InceptionV3(weights='imagenet', include_top=False)

    else:
        base_model = InceptionV3(include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(params['dropout'])(x)
    predictions = Dense(params['nb_classes'], activation=params['activation_function'])(x)
    '''
    for layer in base_model.layers:
        layer.trainable = False
    '''
    model = Model(inputs=base_model.input, outputs=predictions)

    if weights:
        model.load_weights(weights)


    return model

def get_dense(params, weights):

    model = Sequential()
    model.add(Dense(params['dense_layer'], input_shape=(params['input_size'],),name='dense_1'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['nb_classes'], name='dense_2'))
    model.add(Activation(params['activation_function']))

    if weights:
        model.load_weights(weights)

    return model


def get_small(params, weights):

    model = Sequential()
    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_a', strides=(1, 1),
                            input_shape=(params['im_size'], params['im_size'], params['nb_channels'])))

    model.add(Convolution2D(64, (3, 3), activation='relu', padding='same', name='conv1_b', strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(96, (3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Convolution2D(128, (3, 3), activation='relu', padding='same', name='conv3', strides=(1, 1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(Convolution2D(160, (3, 3), activation='relu', padding='same', name='conv4', strides=(1, 1)))
    model.add(MaxPooling2D(pool_size=(3, 3)))

    model.add(Flatten())
    model.add(Dense(1024, name='dense_1'))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(Dense(params['nb_classes'], name='dense_2'))
    model.add(Activation(params['activation_function']))

    if weights:
        model.load_weights(weights)

    return model


def get_vgg16(pre_trained, params, weights):
    if pre_trained:
        base_model = VGG16(weights='imagenet', include_top=False)

    else:
        base_model = VGG16(include_top=False)

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    x = Dropout(params['dropout'])(x)
    predictions = Dense(params['nb_classes'], activation=params['activation_function'])(x)

    for layer in base_model.layers:
        layer.trainable = False

    model = Model(inputs=base_model.input, outputs=predictions)

    if weights:
        model.load_weights(weights)

    return model


def create_top(model, params):
    from keras.layers.core import Dense, Dropout

    for layer in model.layers:
        # Turn layers trainable False.
        layer.trainable = False

    for i in range(5):
        # Remove top layers.
        model.layers.pop()

    # Add new layers.
    model.add(Dense(1024, activation='relu', name='fc6_2'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(1024, activation='relu', name='fc7_2'))
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['nb_classes'], activation=params['activation_function'], name='fc8_2'))

    return model



def get_c3d(pre_trained, params, weights=None):
    from keras.layers import Conv3D
    from keras.layers import MaxPooling3D
    from keras.layers import ZeroPadding3D
    from keras.models import model_from_json
    from keras.layers.normalization import BatchNormalization

    input_shape=(params['volume'], params['im_size'], params['im_size'], params['nb_channels'])#112, 112, 3) # l, h, w, c

    model = Sequential()
    # 1st layer group
    model.add(Conv3D(64, (3, 3, 3), activation='relu',
                     padding='same', name='conv1',
                     strides=(1, 1, 1),
                     input_shape=input_shape))#(params['im_size'], params['im_size'], params['volume'], params['nb_channels'])))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2),
                           padding='valid', name='pool1'))
    # 2nd layer group
    model.add(Conv3D(128, (3, 3, 3), activation='relu',
                     padding='same', name='conv2',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool2'))
    # 3rd layer group
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3a',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Conv3D(256, (3, 3, 3), activation='relu',
                     padding='same', name='conv3b',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())

    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool3'))
    # 4th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4a',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv4b',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool4'))
    # 5th layer group
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5a',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(Conv3D(512, (3, 3, 3), activation='relu',
                     padding='same', name='conv5b',
                     strides=(1, 1, 1)))
    model.add(BatchNormalization())
    model.add(ZeroPadding3D(padding=(0, 1, 1)))
    model.add(MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           padding='valid', name='pool5'))
    model.add(Flatten())
    # FC layers group
    model.add(Dense(1024, activation='relu', name='fc6'))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(1024, activation='relu', name='fc7'))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))
    model.add(Dense(params['nb_classes'], activation=params['activation_function'], name='fc8'))

    if pre_trained:
        model.load_weights(weights)
        model = create_top(model, params)

    return model

def get_alex(params, weights=None):

    #inputs = Input(shape=(params['nb_channels'], params['im_size'], params['im_size']))

    inputs = Input(shape=(3, params['im_size'], params['im_size']))

    #model.add(Convolution2D(96, (3, 3), activation='relu', padding='same', name='conv2', strides=(1, 1)))

    conv_1 = Convolution2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(inputs)

    conv_2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_1)
    conv_2 = crosschannelnormalization(name='convpool_1')(conv_2)
    conv_2 = ZeroPadding2D((2, 2))(conv_2)
    conv_2 = merge([
                       Convolution2D(128, 5, 5, activation='relu', name='conv_2_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_2)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_2')

    conv_3 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(conv_2)
    conv_3 = crosschannelnormalization()(conv_3)
    conv_3 = ZeroPadding2D((1, 1))(conv_3)
    conv_3 = Convolution2D(384, 3, 3, activation='relu', name='conv_3')(conv_3)

    conv_4 = ZeroPadding2D((1, 1))(conv_3)
    conv_4 = merge([
                       Convolution2D(192, 3, 3, activation='relu', name='conv_4_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_4)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_4')

    conv_5 = ZeroPadding2D((1, 1))(conv_4)
    conv_5 = merge([
                       Convolution2D(128, 3, 3, activation='relu', name='conv_5_' + str(i + 1))(
                           splittensor(ratio_split=2, id_split=i)(conv_5)
                       ) for i in range(2)], mode='concat', concat_axis=1, name='conv_5')

    dense_1 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='convpool_5')(conv_5)

    dense_1 = Flatten(name='flatten')(dense_1)
    dense_1 = Dense(4096, activation='relu', name='dense_1')(dense_1)
    dense_2 = Dropout(0.5)(dense_1)
    dense_2 = Dense(4096, activation='relu', name='dense_2')(dense_2)
    dense_3 = Dropout(0.5)(dense_2)
    dense_3 = Dense(1000, name='dense_3')(dense_3)
    prediction = Activation('softmax', name='softmax')(dense_3)

    model = Model(input=inputs, output=prediction)

    if weights:
        model.load_weights(weights)

    return model


def get_lstm(params, weights, summary=False):
    """
    Sepp Hochreiter Jurgen Schmidhuber (1997). "Long short-term memory".
    Neural Computation. 9 (8): 1735–1780. doi:10.1162/neco.1997.9.8.1735.

    Yue-Hei Ng, Joe, et al.
    "Beyond short snippets: Deep networks for video classification."
    Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
    https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Ng_Beyond_Short_Snippets_2015_CVPR_paper.pdf
    """

    model = Sequential()

    # model.add(recurrent.GRU(params['nb_hidden'], return_sequences=True,
    #                         input_dim=params['input_dim'],
    #                         input_length=params['input_size']))
    # model.add(BatchNormalization())


    # model.add(recurrent.GRU(params['nb_hidden'], return_sequences=True,
    #                         input_dim=params['input_dim'],
    #                         input_length=params['input_size']))
    # model.add(BatchNormalization())

    # model.add(recurrent.GRU(params['nb_hidden'], return_sequences=True,
    #                          input_dim=params['input_dim'],
    #                          input_length=params['input_size']))
    # model.add(BatchNormalization())

    # model.add(recurrent.GRU(params['nb_hidden'], return_sequences=True,
    #                          input_dim=params['input_dim'],
    #                          input_length=params['input_size']))
    # model.add(BatchNormalization())

    model.add(recurrent.GRU(params['nb_hidden'], return_sequences=True,
                            input_dim=params['input_dim'],
                            input_length=params['input_size']))
    model.add(BatchNormalization())
    model.add(Dropout(params['dropout']))

    model.add(TimeDistributed(Dense(params['nb_classes'])))
    model.add(Activation('softmax'))

    if summary:
        print(model.summary())

    if weights:
        model.load_weights(weights)

    return model
