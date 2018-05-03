#!/bin/bash
#
#  predictor.sh
#
# This script will execute all the necessary predictions with predict.py
# @author:Jrz
#----------------------------------------------------------------------------

# Network
VGG="vgg16"
V3="v3"

# Number of classes
DOG=10
KSCGR=9
UCF=11

# Path to the means
OFL_KSCGR_MEAN="/home/trainman/Documents/jrz/TemporalNetV1/Data/kscgr/kscgr_hof_mean.png"
RGB_KSCGR_MEAN="/home/trainman/Documents/jrz/TemporalNetV1/Data/kscgr/kscgr_mean.png"
RGB_DOG_MEAN="/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/mean.png"
RGB_UCF_MEAN="/home/trainman/Documents/jrz/UCF11/mean/trainval_mean.png"

# Path to the tests
OFL_KSCGR_TEST="/home/trainman/Documents/KSCGR_HOF/path_67.txt"
RGB_KSCGR_TEST="/home/trainman/Documents/KSCGR/path_67.txt"
RGB_DOG_TEST="/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/test.txt"
RGB_UCF_TEST="/home/trainman/Documents/jrz/UCF11/test.txt"


# Paths to the weights
V3_OFL_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/kitchen_of/0.5/0.005/weights.hdf5"
VGG16_OFL_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/kitchen_of/0.5/0.005/weights.hdf5"
VGG16_RGB_DOG="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/dog_rgb/0.7_best/0.001/weights.hdf5"
V3_RGB_DOG="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/dog_rgb/0.7_BEST/0.001/weights.hdf5"
V3_RGB_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/kitchen_rgb/new/0.9/0.005/weights.hdf5"
VGG16_RGB_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/kitchen_rgb/0.5/0.001/weights.hdf5"

V3_RGB_UCF="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/ucf_rgb/0.95/0.001/weights.hdf5"
VGG16_RGB_UCF="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/ucf_rgb/0.9/0.005/weights.hdf5"


# Example
# python predict.py ${NAME_OF_THE_NETWORK} ${WEIGHTS_OF_THE_MODEL} ${PATH_TO_THE_TEST_FILE} ${FILE_TO_THE_MEAN} ${NUMBER_OF_CLASSES}

#python predict.py ${V3} ${V3_OFL_KSCGR} ${OFL_KSCGR_TEST} ${OFL_KSCGR_MEAN} ${KSCGR}
#python predict.py ${VGG} ${VGG16_OFL_KSCGR} ${OFL_KSCGR_TEST} ${OFL_KSCGR_MEAN} ${KSCGR}
#python predict.py ${VGG} ${VGG16_RGB_KSCGR} ${RGB_KSCGR_TEST} ${RGB_KSCGR_MEAN} ${KSCGR}

python predict.py ${V3} ${V3_RGB_UCF} ${RGB_UCF_TEST} ${RGB_UCF_MEAN} ${UCF}
python predict.py ${VGG} ${VGG16_RGB_UCF} ${RGB_UCF_TEST} ${RGB_UCF_MEAN} ${UCF}

#python predict.py ${V3} ${V3_RGB_KSCGR} ${RGB_KSCGR_TEST} ${RGB_KSCGR_MEAN} ${KSCGR}
#python predict.py ${VGG} ${VGG16_RGB_KSCGR} ${RGB_KSCGR_TEST} ${RGB_KSCGR_MEAN} ${KSCGR}


