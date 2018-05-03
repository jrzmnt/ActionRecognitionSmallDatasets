#!/usr/bin/python
#-*- coding: utf-8 -*-

import linecache
import logging
import sys
logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
from collections import OrderedDict, defaultdict
import ast
import operator
import numpy as np
from os.path import realpath, dirname, join
#from os.path import join, exists

import progressbar

def extract_vectors(inputfile, with_paths=False):
    """
    Generates the training vectors from the input file

    Parameters
    ----------
    inputfile : string
        path to the file containing the frame, the true label and a set of features
    with_paths : boolean
        returns the vector containing the path of each image (True|False)

    Returns
    -------
    paths : array_like
        list containing all paths of images
    y : array_like
        list containing all the true labels
    X : array_like
        list with a list of all features [[f1.1,f1.2],[f2.1,f2.2]] 
    """
    paths, X, y = [], [], []
    logger.info('processing test images from %s' % inputfile)
    pf = FileOfPaths(inputfile)
    if not pf.has_features():
        logger.error('Cannot find features in file: %s' % inputfile)
        sys.exit(0)

    pb = progressbar.ProgressBar(pf.nb_lines)
    for path, label, feats in pf:
        if with_paths:
            paths.append(path)
        X.append(feats)
        y.append(label)
        pb.update()
    X = np.array(X).astype(float)
    y = np.array(y).astype(int)
    return paths, y, X


def bin2matrix(filename):
    """
    Load numpy binary file and return matrices `X` and `y`
    """
    filename = realpath(filename)
    Xy = np.load(filename)
    if len(Xy.shape) == 3:
        X = Xy[:,:,:-1]
        y = Xy[:,:,-1]
    else:
        X = Xy[:,:-1]
        y = Xy[:,-1]
    return X, y


def matrix2bin(X, y, filename):
    """
    Save matrices `X` and `y` in `filename`
    """
    if len(X.shape) == 3:
        Xy = []
        for dimX, dimy in zip(X, y):
            ym = np.array([dimy])
            Xy.append(np.append(dimX, ym.T, axis=1))
        Xy = np.array(Xy)
    else:
        ym = np.array([y])
        Xy = np.append(X, ym.T, axis=1)
    np.save(filename, Xy)

def load_features(inputfile, load_bin=False, save_path=None):
    """
    Create `X` and `y` matrices from file

    Parameters:
    -----------
    inputfile : string
        path to the file containing the frame, the true label and the list of features
    """
    if load_bin:
        return bin2matrix(inputfile)
    X, y = [], []

    pf = FileOfPaths(inputfile)

    #pb = progressbar.ProgressBar(pf.numberFiles())
    for n, _ in enumerate(pf):
        y.append(pf.getY())
        X.append(pf.getFeatures())
        #pb.update()
    X = np.array(X).astype(float)
    y = np.array(y).astype(int)
    if save_path:
        ym = np.array([y])
        Xy = np.append(X, ym.T, axis=1)
        np.save(save_path, Xy)
    return X, y

def create_matrix_lstm(inputfile, frames=32, stride=16, load_bin=False, save_in=None):
    """
    From a file containing paths true labels and features creates a matrix with dimensions as batches
    having `frames` elements.

    Parameters:
    -----------
    inputfile : string
        path to the file containing images, true labels and features
    frames : int
        number of frames in each batch
    stride : int
        number of frames that the batch will skip
    load_bin : binary
        True : load `inputfile` as a binary file
        False : load `inputfile` as text file
    save_in : {string,None}
        `string` : path to the file where the matrix will be saved
        None : do not save matrices

    Example:
    --------
    From a file containing:
        path1 0 0.1 0.1 0.1
        path2 0 0.2 0.2 0.2
        path3 0 0.3 0.3 0.3
        path4 0 0.4 0.4 0.4
    And `frames=2`, `stride=1`, generates matrices X and y as:
        X = [
             [[0.1, 0.1, 0.1],
              [0.2, 0.2, 0.2]],
             [[0.2, 0.2, 0.2],
              [0.3, 0.3, 0.3]],
             [[0.3, 0.3, 0.3],
              [0.4, 0.4, 0.4]]
            ]
        y = [[0, 0], [0, 0], [0, 0]]

        X.shape (dimension, frames, features): (3, 2, 3)
        y.shape (dimension, frames) : (3, 2)
    """
    inputfile = realpath(inputfile)
    dirin = dirname(inputfile)

    logger.info('Loading data from %s' % inputfile)
    if load_bin:
        return load_features(inputfile, load_bin=load_bin)

    X, y = load_features(inputfile, load_bin=load_bin)
    # fill missing rows
    nb_rows, nb_cols = X.shape



    addframes = stride - ((nb_rows - frames) % stride)

    logger.info('Adding %d frames to the end of the matrix' % addframes)
    for i in range(addframes):
        X = np.vstack([X, X[-1]])
        y = np.hstack([y, y[-1]])

    logger.info('Building matrix...')
    nb_rows, nb_cols = X.shape
    nb_dims = 1 + ((nb_rows - frames) / stride)
    logger.info('Dividing into %d dimensions' % nb_dims)
    X_total, y_total = [], []
    index = 0
    for dim in xrange(nb_dims):
        final = index + frames
        X_total.append(X[index:final])
        y_total.append(y[index:final])
        index = index + stride
    X = np.array(X_total)
    y = np.array(y_total)

    logger.info('New shape X (dim, frames, feats): %s' % str(X.shape))
    logger.info('New shape y (dim, frames): %s' % str(y.shape))

    if save_in:
        logger.info('Saving file: %s' % save_in)
        matrix2bin(X, y, save_in)
    return X, y


def extract_matrix_lstm(inputfile, nb_regions=1, load_bin=False, save_bin=None):
    """
    From a file containing paths true labels and features creates a matrix 
    with dimensions as batches having `nb_regions` elements.

    Parameters
    ----------
    inputfile : string
        path to the file containing images, true labels and features
    nb_regions : int
        number of regions in each batch
    stride : int
        number of frames that the batch will skip
    load_bin : binary
        True : load `inputfile` as a binary file
        False : lzoad `inputfile` as text file
    save_bin : {string,None}
        `string` : path to the file where the matrix will be saved
        None : do not save matrices 

    Notes
    -----
    The input file must contain `path`, `true label` and `features` of all regions
    Please run `temporal.py` with the number of regions and save the content in a 
    file before passing it as `inputfile` to this function.

    Example:
    --------
    From a file containing 2 regions:
        path1 0 0.1 0.1 0.1 0.4 0.4 0.4
        path2 0 0.2 0.2 0.2 0.3 0.3 0.3
        path3 0 0.1 0.1 0.1 0.3 0.3 0.3
        path4 0 0.2 0.2 0.2 0.4 0.4 0.4
    And `nb_regions=2`, `stride=1`, generates matrices X and y as:
        X = [
             [[0.1, 0.1, 0.1], 
              [0.4 0.4 0.4]],
             [[0.2, 0.2, 0.2], 
              [0.3, 0.3, 0.3]],
             [[0.1, 0.1, 0.1], 
              [0.3, 0.3, 0.3]]
             [[0.2, 0.2, 0.2], 
              [0.4, 0.4, 0.4]]
            ]
        y = [[0, 0], [0, 0], [0, 0], [0, 0]]

        X.shape (dimension, regions, features): (4, 2, 3)
        y.shape (dimension, regions) : (4, 2)
    """
    inputfile = realpath(inputfile)
    dirin = dirname(inputfile)

    logger.info('Loading data from %s' % inputfile)
    if load_bin:
        return bin2matrix(inputfile)

    _, y, X = extract_vectors(inputfile, with_paths=False)
    nb_rows, nb_feats = X.shape
    single = nb_feats / nb_regions
    X_dim, y_dim = [], []
    for row, label in zip(X, y):
        X_dim.append(np.reshape(row, (nb_regions, single)))
        y_dim.append([label]*nb_regions)
    X = np.array(X_dim).astype(float)
    y = np.array(y_dim).astype(int)
    X_dim, y_dim = [], []

    logger.info('New shape X (dim, frames, feats): %s' % str(X.shape))
    logger.info('New shape y (dim, frames): %s' % str(y.shape))

    if save_bin:
        logger.info('Saving file: %s' % save_bin)
        matrix2bin(X, y, save_bin)
    return X, y


def createDictionaryFromFile(inputfile):
    """
    Create a dictionary with the path of the input file. The path are
    represented as a fixed structure from <HOME> to <image.jpg> as:

    ```
    <HOME>/data1/scramble-egg/img256/<image.jpg> 0
    ```

    The dictionary is structured as follows:
        {id_data:
            {activity:
                {size:
                    {id_img: [y, pred]
                    }
                }
            }
        }

    Parameters:
    -----------
    inputfile : string
        path to the file containing paths and labels

    Returns:
    --------
    home : string
        path to the <HOME> location
    dic : dict
        dictionary containing the path structure and labels
    """
    logger.info('loading file: %s' % inputfile)
    dic = {}
    with open(inputfile) as fin:
        for n, line in enumerate(fin, start=1):
            arr = line.strip().split()
            path = arr[0]

            labels = []
            for label in arr[1:]:
                labels.append(ast.literal_eval(label))

            cpath = path.split('/')
            id_img = int(cpath[-1].replace('.jpg', ''))
            size_img = cpath[-2]
            activity = cpath[-3]
            id_data = int((cpath[-4])[-1])
            home = '/'.join(cpath[:-4])

            if dic.has_key(id_data):
                if dic[id_data].has_key(activity):
                    if dic[id_data][activity].has_key(size_img):
                        dic[id_data][activity][size_img][id_img] = labels
                    else:
                        dic[id_data][activity][size_img] = {id_img: labels}
                else:
                    dic[id_data][activity] = {size_img: {id_img: labels}}
            else:
                dic[id_data] = {activity: {size_img: {id_img: labels}}}
    return n, home, dic


def createPathY(inputfile):
    """
    Create a dictionary containing only the path of the image and the true label

    Parameters:
    -----------
    inputfile : string
        Path to the file containing images and true labels
    """
    dic = OrderedDict()
    with open(inputfile) as fin:
        for n, line in enumerate(fin, start=1):
            arr = line.strip().split()
            path, lbls = arr[0], arr[1:]
            dic[path] = lbls
            if n == 1:
                home = dirname(path)
    return n, home, dic

class PathFile(object):
    """
    Class to extract the content from the file containing paths,
    true labels and predicted labels.
    """

    def __init__(self, inputfile, imlist=True, use_dic=True):
        """
        Initiate the PathFile class

        Parameters:
        -----------
        inputfile : string
            Path to the file containing paths and labels
        imlist : boolean
            True : Yield the list of images containing in image folder
            False: Yield an image at time
        use_dic : boolean
            Build a dictionary containing each activity separately
        """
        self.inputfile = inputfile
        self.imlist = imlist
        self.use_dic = use_dic
        if use_dic:
            self.n, self.home, self.dic = createDictionaryFromFile(inputfile)
        else:
            self.n, self.home, self.dic = createPathY(inputfile)
        self.data = 0
        self.activity = None
        self.imsize = None
        self.img = 0
        self.i = 0
        self.labels = []
        self.allimgs = []
        self.alllabels = []
        self.fout = None
        self.record_mode = False

    def __iter__(self):
        """
        Iterate over the file yielding an activity each time
        """
        if self.use_dic:
            for data in sorted(self.dic):
                self.data = data
                for activity in sorted(self.dic[data]):
                    self.activity = activity
                    for imsize in sorted(self.dic[data][activity]):
                        self.imsize = imsize
                        self.allimgs, self.alllabels = [], []
                        for img in sorted(self.dic[data][activity][imsize]):
                            self.img = img
                            self.labels = self.dic[data][activity][imsize][img]
                            if self.imlist:
                                self.allimgs.append(self.img)
                                self.alllabels.append(self.labels)
                            else:
                                yield data, activity, imsize, img, self.labels
                                self.i += 1
                        if self.imlist:
                            yield data, activity, imsize, self.allimgs, self.alllabels
                            self.i += 1
        else:
            for data in sorted(self.dic):
                self.img = data
                self.labels = self.dic[data]
                yield self.img, self.labels
                self.i += 1

    def numberFiles(self):
        """
        Return the total number of images in the input file
        """
        return self.n

    def numberActivities(self):
        """
        Return the number of activities in dataset
        """
        if self.use_dic:
            nb_data = self.dic.keys()
            nb_act = (self.dic[nb_data[0]]).keys()
            return len(nb_data) * len(nb_act)
        else:
            return -1

    def imagePath(self):
        """
        Return the image path of the current line being processed
        """
        if self.use_dic:
            if self.imlist:
                paths = []
                for img in self.allimgs:
                    paths.append(
                        join(self.home, 'data' + str(self.data), self.activity, self.imsize, str(img) + '.jpg'))
                return paths
            else:
                path = join(self.home, 'data' + str(self.data), self.activity, self.imsize, str(self.img) + '.jpg')
        else:
            path = self.img
        return path

    def getActivity(self):
        """
        Return the label of the current activity
        """
        return self.activity

    def getIdData(self):
        """
        Return the label of the current activity
        """
        return self.data

    def localPath(self):
        """
        Return the path up to the <HOME> folder
        """
        return self.home

    def getType(self):
        """
        Return the type of images being processed.
        Types:
            rgb : still images
            hof : optical flow
            bgs : background subtraction
        """
        if self.use_dic:
            data = self.dic.keys()[0]
            act = self.dic[data].keys()[0]
            return self.dic[data][act].keys()[0]
        else:
            return None

    def getY(self):
        """
        Return the true label
        """
        return self.labels[0]

    def getCounter(self):
        """
        Return the counter of images
        """
        return self.i

    def getFirst(self):
        """
        Return the first image and path
        """
        if self.use_dic:
            data = sorted(self.dic.keys())[0]
            activity = sorted(self.dic[data].keys())[0]
            imsize = sorted(self.dic[data][activity].keys())[0]
            img = sorted(self.dic[data][activity][imsize].keys())[0]
            labels = self.dic[data][activity][imsize][img]
            path = join(self.home, 'data' + str(data), activity, imsize, str(img) + '.jpg')
        else:
            first = self.dic.keys()[0]
            path, labels = first, self.dic[first]
        return path, labels

    def getYandPred(self):
        """
        Return the list of true labels and a list of the predicted labels.
        This list refers to a certain data, activity and imsize.

        Returns:
        --------
        y : array_like
            list contanining the true labels
        pred : array_like
            list contanining the predicted labels
        """
        y = []
        pred = []
        if self.use_dic:
            for k in self.alllabels:
                if len(k) == 2:
                    y.append(k[0])
                    pred.append(k[1])
                else:
                    return None
        else:
            for k in self.dic.values():
                if len(k) == 2:
                    y.append(k[0])
                    pred.append(k[1])
                else:
                    return None
        return y, pred

    def allYandPred(self):
        """
        Return the list of true labels and a list of the predicted labels.
        This list refers to all elements in the file.

        Returns:
        --------
        y : array_like
            list contanining the true labels
        pred : array_like
            list contanining the predicted labels
        """
        y = []
        pred = []
        if self.use_dic:
            pb = progressbar.ProgressBar(self.n)
            for data in sorted(self.dic):
                for activity in sorted(self.dic[data]):
                    for imsize in sorted(self.dic[data][activity]):
                        for img in self.dic[data][activity][imsize]:
                            labels = self.dic[data][activity][imsize][img]
                            if len(labels) == 2:
                                y.append(labels[0])
                                pred.append(labels[1])
                            else:
                                return None
                            pb.update()
        else:
            y, pred = self.getYandPred()
        return y, pred

    def getDataset(self, data):
        """
        Return the `data` dataset from `self.dic`

        Parameters:
        -----------
        data : int
            integer corresponding to the id of the dataset

        Returns:
        --------
        dic : dict
            dictionary of the corresponding dataset
        """
        if self.dic.has_key(data):
            return self.dic[data]
        else:
            return None

    def imgOutputPath(self, newpath):
        """
        Return the new output path to each file based on the new path.
        Substitute ``self.home'' by the ``newpath''.

        Parameters:
        -----------
        newpath : string
            path to the folder that substitutes ``self.home''
        """
        newimg = self.imagePath().replace(self.home, newpath)
        return newimg

    def getFeatures(self, N=2):
        """
        Return a list containing the `N` features with the highest scores

        Parameters:
        -----------
        N : int
            number of features to return

        Notes:
        ------
        The output has the form of a list as:
        [(1, 0.33), (3, 0.12), (5, 0.08), ...]
        """
        features = self.labels[1:]
        classes = map(int, features[0::2])
        preds = np.array(features[1::2], dtype=np.float32)
        topN = []
        for n in range(N):
            valmax = preds.max()
            imax = preds.argmax()
            topN.append((classes[imax], valmax))
            preds[imax] = -1
        return topN

    def saveNewData(self, newvec, fname=None):
        """
        Save a new list of elements (usually predicted labels changed
        by the window filter

        Parameters:
        -----------
        newvec : array_like
            list containing predicted labels to record in the file

        Notes:
        ------
        This method must be used with `imlist=True`
        """
        assert len(newvec) == len(self.allimgs), \
            'List is uncompatible with images %d != %d' % (len(newvec), len(self.allimgs))
        if fname and not self.record_mode:
            self.fout = open(fname, 'w')
            self.record_mode = True
        vecY, _ = self.getYandPred()
        for img, y, pred in zip(self.allimgs, vecY, newvec):
            path = join(self.home, 'data' + str(self.data), self.activity, self.imsize, str(img) + '.jpg')
            self.fout.write('%s %d %d\n' % (path, y, pred))

    def close(self):
        """
        Close whatever is opened
        """
        if self.record_mode:
            self.fout.close()
            self.record_mode = False
            # End of class PathFile


class FileOfPaths(object):
    """
    Class to deal with generic file containing paths and true labels
    """

    def __init__(self, inputfile):
        """
        Initialize the class

        Parameters:
        -----------
        inputfile : string
            path to the file containing images and true labels
        """
        self.inputfile = realpath(inputfile)
        self.dirin = dirname(inputfile)
        self.nb_lines = self.count_lines(inputfile)
        self.n = 0
        self.path = None
        self.y = None
        self.feats = None
        self.ally = None
        self.allimgs = []
        self.record_mode = False

    def __iter__(self):
        """
        Iterates the file yielding the path, the true label and a vector of
        features when exists
        """
        import sys
        with open(self.inputfile) as fin:
            for line in fin:
                arr = line.strip().split()
                self.path = arr[0]                
                self.y = int(arr[1])
                #sys.exit(0)
                if len(arr) > 2:
                    self.feats = map(float, arr[2:])
                    yield self.path, self.y, self.feats
                else:
                    yield self.path, self.y

    @staticmethod
    def count_lines(inputfile):
        """ Returns the total number of images in the input file
        Parameters
        ----------
        inputfile : string
            path to the file containing images and labels/features
        Returns
        -------
        n : int
            total number of lines in the document
        """
        with open(inputfile) as fin:
            for n, _ in enumerate(fin, start=1): pass
        return n

    def numberFiles(self):
        """
        Return the total number of images in the input file
        """
        with open(self.inputfile) as fin:
            for n, _ in enumerate(fin, start=1): pass
        self.n = n
        return self.n

    def imagePath(self):
        """
        Return the image path of the current line being processed
        """
        return self.path

    def getY(self):
        """
        Return the true label
        """
        return self.y

    def softmax(self):
        """
        Return the softmax items of the current line

        Softmax is identified from the 3th element of the line as:
            path true_label softmax_1 softmax_2 softmax_3 ...
        """
        return self.feats

    def softmax2feats(self, softmax, sort_values=False):
        """
        Transform softmax into a list containing the index and the probability
        in each class

        Parameters:
        -----------
        softmax : array_like
            list containing the probabilities of each class
        """
        softlist = [(index, val) for index, val in enumerate(softmax)]
        if sort_values:
            return sorted(dict(softlist).items(), key=operator.itemgetter(1), reverse=True)
        return softlist

    def getFeatures(self, N=None, indexes=False):
        """
        Return the N features
        In case of `N != None`, features return with the corresponding
        index

        Parameters:
        -----------
        N : int
            number of features to return

        Example:
        >>>fop = FileOfPaths()
        >>>fop.feats = [0.2, 0.5, 0.05, 0.25]
        >>>vec = fop.getFeatures(N=2)
        >>>[(1, 0.5), (3, 0.25)]
        """
        if indexes:
            features = self.softmax2feats(self.feats, sort_values=True)
        else:
            features = self.feats
        if N:
            return features[:N]
        return features

    def allYandPred(self):
        """
        Return the list of true labels and a list of the predicted labels.
        This list refers to all elements in the file.

        Returns:
        --------
        y : array_like
            list contanining the true labels
        pred : array_like
            list contanining the predicted labels
        """
        y = []
        pred = []
        with open(self.inputfile) as fin:
            for line in fin:
                arr = line.strip().split()
                label, feat = int(arr[1]), int(arr[2])
                self.allimgs.append(arr[0])
                y.append(label)
                pred.append(feat)
        self.ally = y
        return y, pred

    def saveNewData(self, newvec, fname=None):
        """
        Save a new list of elements (usually predicted labels changed
        by the window filter

        Parameters:
        -----------
        newvec : array_like
            list containing predicted labels to record in the file

        Notes:
        ------
        This method must be used with `imlist=True`
        """
        assert len(newvec) == len(self.allimgs), \
            'List is uncompatible with images %d != %d' % (len(newvec), len(self.allimgs))
        if fname and not self.record_mode:
            self.fout = open(fname, 'w')
            self.record_mode = True
        for path, y, pred in zip(self.allimgs, self.ally, newvec):
            self.fout.write('%s %d %d\n' % (path, y, pred))

    def classesAndFrames(self):
        """
        Return all classes along with the number of frames
        from the inputfile
        """
        classes = defaultdict(int)
        with open(self.inputfile) as fin:
            for line in fin:
                arr = line.strip().split()
                y = int(arr[1])
                classes[y] += 1
        return classes

    def allElementsFromClass(self, c, inverse=False):
        """
        Return all paths that belong to the class `c`.

        Parameters
        ----------
        c : int
            class to extract the content
        inverse : boolean
            True : return frames of all classes but `c`
            False : return frames of `c`
        """
        frames = []
        with open(self.inputfile) as fin:
            for line in fin:
                arr = line.strip().split()
                y = int(arr[1])
                if not inverse and y == c:
                    frames.append(arr[0])
                elif inverse and y != c:
                    frames.append(arr[0])
        return frames

    def numberOfClasses(self):
        """
        Return the number of classes from the inputfile
        """
        classes = self.classesAndFrames()
        return len(classes.keys())

    def content(self):
        """
        Return the content of a file as list of tuples `(path, true_label)`
        """
        with open(self.inputfile) as fin:
            cont = []
            for line in fin:
                arr = line.strip().split()
                y = int(arr[1])
                cont.append((arr[0], y))
        return cont

    def close(self):
        """
        Close whatever is opened
        """
        if self.record_mode:
            self.fout.close()
            self.record_mode = False
            # End of class FileOfPaths

    def get_line(self, nb_line):
        """Returns the line at number `nb_line`"""
        return linecache.getline(self.inputfile, nb_line).strip()

    def get_path(self, nb_line, label=False):
        """Returns the path of the line at number `nb_line`"""
        line = self.get_line(nb_line)
        path = None
        if line:
            arr = line.split()
            path = arr[0]
            if label:
                y = arr[1]
                return path, y
        return path

    def features_line(self, nb_line, asstr=False):
        """Returns only the feature of the line number `nb_line`"""
        line = linecache.getline(self.inputfile, nb_line)
        arr = line.strip().split()
        if len(arr) > 2:
            if asstr:
                return ' '.join(arr[2:])
            else:
                return arr[2:]
        return None

    def has_features(self):
        """
        Returns
        -------
        _ : bool
            False : only path and label
            True : path, label and features
        """
        line = self.get_line(1)
        arr = line.strip().split()
        if len(arr) > 2:
            return True
        else:
            return False
