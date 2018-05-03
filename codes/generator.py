# coding=utf-8

import random
import cv2
from keras.preprocessing.image import *
from keras import backend as K
import logging
logger = logging.getLogger('models.keras.preprocessing')
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
handler = logging.FileHandler('debug.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

def create_gen():
    """
        Create a generator for train and test sets.
        
        :type data_path: string
        :param data_path: path to the dataset with the following format: path/to/image class
        :rtype: ImageDataGenerator object
        :return: a generator to images
    """

    datagen = ImageFileGenerator(
        rescale=1./255
    )

    return datagen


class CreateTemporalGen(object):
    def __init__(self, data_path, params):
        self.lines = open(data_path, 'r').readlines()
        self.params = params
        self.id_reader = 0
        self.nb_sample = len(self.lines)
        self.n_features = len(self.lines[0].split()[2:])

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):

        batch_x = np.zeros((self.params['batch_size'], self.n_features))
        batch_y = np.zeros((self.params['batch_size'], self.params['nb_classes']), dtype='float32')

        for batch_num in range(self.params['batch_size']):

            # Divide path, class, and features.
            y, features = self.lines[self.id_reader].split()[1], self.lines[self.id_reader].split()[2:]

            # Increase reader.
            self.id_reader += 1

            if self.id_reader == self.nb_sample:
                self.id_reader = 0

            # Add to batch.
            batch_x[batch_num] = np.array(features)

            # Add class.
            batch_y[batch_num, int(y)] = 1.

        return batch_x, batch_y


class CreateGen3D(object):
    def __init__(self, data_path, params):
        self.lines = open(data_path, 'r').readlines()
        self.params = params
        self.id_reader = 0
        self.nb_sample = len(self.lines)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def next(self):
        
        batch_x = np.zeros((self.params['batch_size'], self.params['volume'], self.params['im_size'], self.params['im_size'], self.params['nb_channels']))
        batch_y = np.zeros((self.params['batch_size'], self.params['nb_classes']), dtype='float32')

        for batch_num in range(self.params['batch_size']):

            frames = np.zeros((self.params['volume'], self.params['im_size'], self.params['im_size'], self.params['nb_channels']))
            for i in range(self.params['volume']):

                # Divide path and class.
                path, y = self.lines[self.id_reader].split()

                # Read image.
                img = cv2.imread(path)
                img = cv2.resize(img,(self.params['im_size'], self.params['im_size']))
                frames[i] = img_to_array(img)

                # Increase reader.
                self.id_reader += 1

                if self.id_reader == self.nb_sample:
                    self.id_reader = 0

            batch_x[batch_num] = frames        
            # Add class.            
            batch_y[batch_num, int(y)] = 1.

        # batch_x = batch_x.reshape((self.params['batch_size'], self.params['im_size'], self.params['im_size'], self.params['volume'], self.params['nb_channels']))
        
        return batch_x, batch_y        

        # else:
        #     raise StopIteration()


class FileImageIterator(Iterator):
    """
    This class is based on `DirectoryIterator` from Keras:
    https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py

    This class creates an iterator on a file containing paths to images and true labels.
    """

    def __init__(self, directory, image_data_generator,
                 target_size=(240, 240), color_mode='rgb',
                 classes=None, class_mode='categorical',
                 batch_size=32, shuffle=True, seed=None,
                 save_to_dir=None, save_prefix='', save_format='jpeg', train=True):

        #logger.info("Start FileImageIterator")

        dim_ordering = K.image_data_format()
        self.directory = directory
        self.image_data_generator = image_data_generator
        self.img_volume = self.image_data_generator.volume
        self.target_size = tuple(target_size)
        if color_mode not in {'rgb', 'grayscale'}:
            raise ValueError('Invalid color mode:', color_mode,
                             '; expected "rgb" or "grayscale".')
        self.color_mode = color_mode
        self.dim_ordering = dim_ordering

        if self.color_mode == 'rgb':
            if self.dim_ordering == 'channels_last':
                self.image_shape = self.target_size + (3,)
            else:
                self.image_shape = (3,) + self.target_size
        else:
            if self.dim_ordering == 'channels_last':
                self.image_shape = self.target_size + (1,)
            else:
                self.image_shape = (1,) + self.target_size
        self.classes = classes
        if class_mode not in {'categorical', 'binary', 'sparse', None}:
            raise ValueError('Invalid class_mode:', class_mode,
                             '; expected one of "categorical", '
                             '"binary", "sparse", or None.')
        self.class_mode = class_mode
        self.save_to_dir = save_to_dir
        self.save_prefix = save_prefix
        self.save_format = save_format

        self.batch_size = batch_size

        # first, count the number of samples and classes
        self.nb_sample = 0
        self.class_indices = {}
        self.filenames = []
        self.classes = []
        with open(self.directory) as fin:

            #logger.info("Reading file.")

            vet = fin.readlines()
            if train:
                random.shuffle(vet)

            for n, line in enumerate(vet, start=1):
                path = line.strip().split()[0]
                y = line.strip().split()[1]

                #if n == 1:
                    #print path, y

                self.classes = np.append(self.classes, int(y))
                self.class_indices[y] = y
                self.filenames.append(path)
                self.nb_sample += 1

        self.nb_class = len(self.class_indices.keys())

        print('Found %d images belonging to %d classes.' % (self.nb_sample, self.nb_class))

        if self.img_volume:
            super(FileImageIterator, self).__init__(self.nb_sample, self.img_volume, shuffle, seed)
        else:
            #print "Is this right?", self.nb_sample, self.batch_size, shuffle, seed
            super(FileImageIterator, self).__init__(self.nb_sample, self.batch_size, shuffle, seed)

    def next(self):

        #print self.directory, self.dim_ordering

        with self.lock:
            # index_array, current_index, current_batch_size = next(self.index_generator)
            index_array = next(self.index_generator)
            current_batch_size = index_array.shape[0]

            # print x

        # The transformation of images is not under thread lock so it can be done in parallel
        if self.img_volume:
            batch_x = np.zeros((current_batch_size, self.img_volume,) + self.image_shape)
        else:
            batch_x = np.zeros((current_batch_size,) + self.image_shape)

        grayscale = self.color_mode == 'grayscale'

        #print self.img_volume

        if self.img_volume:
            for ind in xrange(self.img_volume):
                frames = np.zeros((self.img_volume,) + self.image_shape)
                for i, j in enumerate(index_array):
                    fname = self.filenames[j]
                    img = load_img(os.path.join(self.directory, fname), target_size=self.target_size)
                    x = img_to_array(img, self.dim_ordering)
                    frames[i] = x

                batch_x[i] = frames
            batch_x.reshape((current_batch_size, self.image_shape[0], self.image_shape[1], self.img_volume, self.image_shape[2]))

        else:
            # build batch of image data
            
            for i, j in enumerate(index_array):
                fname = self.filenames[j]
                img = load_img(os.path.join(self.directory, fname), target_size=self.target_size)
                x = img_to_array(img, self.dim_ordering)
                #x = self.image_data_generator.random_transform(x)
                x -= self.image_data_generator.mean
                #x = np.absolute(x)
                x = self.image_data_generator.standardize(x)
                batch_x[i] = x
                #import cPickle
                #cPickle.dump(x, open('img_after_preprocess' + '.p', 'wb'))
            # optionally save augmented images to disk for debugging purposes

        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=self.batch_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))

        if self.class_mode == 'sparse':
            batch_y = self.classes[index_array]

        elif self.class_mode == 'binary':
            batch_y = self.classes[index_array].astype('float32')

        elif self.class_mode == 'categorical':
            batch_y = np.zeros((len(batch_x), self.nb_class), dtype='float32')
            for i, label in enumerate(self.classes[index_array]):
                batch_y[i, int(label)] = 1.
        else:
            return batch_x

        #logger.info("Batch shape: {}\nBatch example: {}".format(batch_x.shape, batch_x[0]))

        return batch_x, batch_y


class ImageFileGenerator(ImageDataGenerator):
    def __init__(self, volume=None, featurewise_center=False, samplewise_center=False,
                 featurewise_std_normalization=False, samplewise_std_normalization=False,
                 zca_whitening=False, rotation_range=0., width_shift_range=0.,
                 height_shift_range=0., shear_range=0., zoom_range=0.,
                 channel_shift_range=0., fill_mode='nearest', cval=0.,
                 horizontal_flip=False, vertical_flip=False, rescale=None,
                 data_format='channels_last'):
        img = ImageDataGenerator.__init__(self,
                                          featurewise_center=featurewise_center,
                                          samplewise_center=samplewise_center,
                                          featurewise_std_normalization=featurewise_std_normalization,
                                          samplewise_std_normalization=samplewise_std_normalization,
                                          zca_whitening=zca_whitening,
                                          rotation_range=rotation_range,
                                          width_shift_range=width_shift_range,
                                          height_shift_range=height_shift_range,
                                          shear_range=shear_range,
                                          zoom_range=zoom_range,
                                          channel_shift_range=channel_shift_range,
                                          fill_mode=fill_mode,
                                          cval=cval,
                                          horizontal_flip=horizontal_flip,
                                          vertical_flip=vertical_flip,
                                          rescale=rescale,
                                          data_format=data_format)

        self.mean = cv2.imread("/home/juarez/TemporalNetV1/Data/dogcentric/dog_centric_mean.png")
        self.volume = volume
        #logger.info("Setting imageFileGenerator")

    def flow_from_file(self, directory, target_size=(240, 240), color_mode='rgb',
                       classes=None, class_mode='categorical', batch_size=32,
                       shuffle=True, seed=None, save_to_dir=None, save_prefix='', save_format='jpeg', train=True):
        # Create flow_from_directory method to yield images from a file
        return FileImageIterator(directory, self, target_size=target_size, color_mode=color_mode,
                                 classes=classes, class_mode=class_mode, batch_size=batch_size, shuffle=shuffle,
                                 seed=seed,
                                 save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format, train=train)

if __name__ == '__main__':

    datagen = create_gen()