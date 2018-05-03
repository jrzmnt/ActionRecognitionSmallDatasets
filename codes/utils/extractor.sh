#!/bin/bash
#
#  extractor.sh
#
# This script will execute all the necessary paths with feature_extractor.py
# @author:Jrz
#----------------------------------------------------------------------------

# Network
VGG="vgg16"
V3="inception_v3"

# Number of classes
DOG=10
KSCGR=9
UCF=11

# Path to the means
OFL_KSCGR_MEAN="/home/trainman/Documents/jrz/TemporalNetV1/Data/kscgr/kscgr_hof_mean.png"
RGB_KSCGR_MEAN="/home/trainman/Documents/jrz/TemporalNetV1/Data/kscgr/kscgr_mean.png"
RGB_DOG_MEAN="/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/mean.png"



# Path to the data
OFL_KSCGR_TRAIN="/home/trainman/Documents/KSCGR_HOF/path_1235.txt"
OFL_KSCGR_VAL="/home/trainman/Documents/KSCGR_HOF/path_4.txt"
OFL_KSCGR_TEST="/home/trainman/Documents/KSCGR_HOF/path_67.txt"

RGB_KSCGR_TRAIN="/home/trainman/Documents/KSCGR/path_1235.txt"
RGB_KSCGR_VAL="/home/trainman/Documents/KSCGR/path_4.txt"
RGB_KSCGR_TEST="/home/trainman/Documents/KSCGR/path_67.txt"

RGB_DOG_TRAIN="/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/train_new.txt"
RGB_DOG_VAL="/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/validation.txt"
RGB_DOG_TEST="/home/trainman/Documents/jrz/TemporalNetV1/Data/dogcentric/test.txt"

# Paths to the weights
V3_OFL_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/kitchen_of/0.5/0.005/weights.hdf5"
VGG16_OFL_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/kitchen_of/0.5/0.005/weights.hdf5"
VGG16_RGB_DOG="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/dog_rgb/0.7_best/0.001/weights.hdf5"
V3_RGB_DOG="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/dog_rgb/0.7_BEST/0.001/weights.hdf5"
V3_RGB_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/kitchen_rgb/new/0.9/0.005/weights.hdf5"
VGG16_RGB_KSCGR="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/kitchen_rgb/0.5/0.001/weights.hdf5"


V3_RGB_UCF="/home/trainman/Documents/jrz/TemporalNetV1/experiments/inception_v3/ucf_rgb/0.95/0.001/weights.hdf5"
VGG_RGB_UCF="/home/trainman/Documents/jrz/TemporalNetV1/experiments/vgg-16/ucf_rgb/0.9/0.005/weights.hdf5"

RGB_UCF_TRAIN="/home/trainman/Documents/jrz/UCF11/train.txt"
RGB_UCF_VAL="/home/trainman/Documents/jrz/UCF11/validation.txt"
RGB_UCF_TEST="/home/trainman/Documents/jrz/UCF11/test.txt"

RGB_UCF_MEAN="/home/trainman/Documents/jrz/UCF11/mean/trainval_mean.png"

python feature_extractor.py $V3 ${V3_RGB_UCF} ${RGB_UCF_TRAIN} ${RGB_UCF_MEAN} ${UCF}
python feature_extractor.py $V3 ${V3_RGB_UCF} ${RGB_UCF_VAL} ${RGB_UCF_MEAN} ${UCF}
python feature_extractor.py $V3 ${V3_RGB_UCF} ${RGB_UCF_TEST} ${RGB_UCF_MEAN} ${UCF}

python feature_extractor.py $VGG ${VGG_RGB_UCF} ${RGB_UCF_TRAIN} ${RGB_UCF_MEAN} ${UCF}
python feature_extractor.py $VGG ${VGG_RGB_UCF} ${RGB_UCF_VAL} ${RGB_UCF_MEAN} ${UCF}
python feature_extractor.py $VGG ${VGG_RGB_UCF} ${RGB_UCF_TEST} ${RGB_UCF_MEAN} ${UCF}

# Example
# python feature_extractor.py ${NAME_OF_THE_NETWORK} ${WEIGHTS_OF_THE_MODEL} ${PATH_TO_THE_FILES} ${FILE_TO_THE_MEAN} ${NUMBER_OF_CLASSES}

# run V3 RGB KSCGR
#python feature_extractor.py $V3 ${V3_RGB_KSCGR} ${RGB_KSCGR_TRAIN} ${RGB_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $V3 ${V3_RGB_KSCGR} ${RGB_KSCGR_VAL} ${RGB_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $V3 ${V3_RGB_KSCGR} ${RGB_KSCGR_TEST} ${RGB_KSCGR_MEAN} ${KSCGR}

# run VGG16 RGB KSCGR
#python feature_extractor.py $VGG ${VGG16_RGB_KSCGR} ${RGB_KSCGR_TRAIN} ${RGB_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $VGG ${VGG16_RGB_KSCGR} ${RGB_KSCGR_VAL} ${RGB_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $VGG ${VGG16_RGB_KSCGR} ${RGB_KSCGR_TEST} ${RGB_KSCGR_MEAN} ${KSCGR}


# run V3 OFL KSCGR
#python feature_extractor.py $V3 ${V3_OFL_KSCGR} ${OFL_KSCGR_TRAIN} ${OFL_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $V3 ${V3_OFL_KSCGR} ${OFL_KSCGR_VAL} ${OFL_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $V3 ${V3_OFL_KSCGR} ${OFL_KSCGR_TEST} ${OFL_KSCGR_MEAN} ${KSCGR}

# run VGG16 OFL KSCGR
#python feature_extractor.py $VGG ${VGG16_OFL_KSCGR} ${OFL_KSCGR_TRAIN} ${OFL_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $VGG ${VGG16_OFL_KSCGR} ${OFL_KSCGR_VAL} ${OFL_KSCGR_MEAN} ${KSCGR}
#python feature_extractor.py $VGG ${VGG16_OFL_KSCGR} ${OFL_KSCGR_TEST} ${OFL_KSCGR_MEAN} ${KSCGR}

# run V3 RGB DOG
#python feature_extractor.py $V3 ${V3_RGB_DOG} ${RGB_DOG_TRAIN} ${RGB_DOG_MEAN} ${DOG}
#python feature_extractor.py $V3 ${V3_RGB_DOG} ${RGB_DOG_VAL} ${RGB_DOG_MEAN} ${DOG}
#python feature_extractor.py $V3 ${V3_RGB_DOG} ${RGB_DOG_TEST} ${RGB_DOG_MEAN} ${DOG}

# run VGG16 RGB DOG
#python feature_extractor.py $VGG ${VGG16_RGB_DOG} ${RGB_DOG_TRAIN} ${RGB_DOG_MEAN} ${DOG}
#python feature_extractor.py $VGG ${VGG16_RGB_DOG} ${RGB_DOG_VAL} ${RGB_DOG_MEAN} ${DOG}
#python feature_extractor.py $VGG ${VGG16_RGB_DOG} ${RGB_DOG_TEST} ${RGB_DOG_MEAN} ${DOG}

# run VGG16 OFL DOG
#python feature_extractor.py $VGG ${VGG16_OFL_DOG} ${OFL_DOG_TRAIN} ${OFL_DOG_MEAN} ${DOG}
#python feature_extractor.py $VGG ${VGG16_OFL_DOG} ${OFL_DOG_VAL} ${OFL_DOG_MEAN} ${DOG}
#python feature_extractor.py $VGG ${VGG16_OFL_DOG} ${OFL_DOG_TEST} ${OFL_DOG_MEAN} ${DOG}

# run V3 OFL DOG
#python feature_extractor.py $V3 ${V3_OFL_DOG} ${OFL_DOG_TRAIN} ${OFL_DOG_MEAN} ${DOG}
#python feature_extractor.py $V3 ${V3_OFL_DOG} ${OFL_DOG_VAL} ${OFL_DOG_MEAN} ${DOG}
#python feature_extractor.py $V3 ${V3_OFL_DOG} ${OFL_DOG_TEST} ${OFL_DOG_MEAN} ${DOG}


