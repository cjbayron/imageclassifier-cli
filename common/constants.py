# -*- coding:utf-8 -*-
"""
Constants
"""

# mode names
TRN_PREP_MODE = 'trn_prep'
TST_PREP_MODE = 'tst_prep'
TRN_MODE = 'trn'
TST_MODE = 'tst'

# location of TFRecords
RECORDS_DIR = 'records'
TF_REC_EXT = '.tfrecords'

# IMAGE PARAMS
IMG_EXT = '.png'
# image dimensions (H, W, C)
IMG_SHAPE = (32, 32, 3)

# TRAINING PARAMS
BATCH_SIZE = 32
LRN_RATE = 0.05
NUM_EPOCHS = 20
# set to same value as NUM_EPOCHS if only a single checkpoint is needed
NUM_EPOCH_BEFORE_CHKPT = 20
NUM_BATCH_BEFORE_PRINT_LOSS = 100

# TESTING PARAMS
# [IMPORTANT: ALWAYS SET TEST BATCH SIZE to a DIVISOR OF NUMBER of TESTING DATA!]
TST_BATCH_SIZE = 100
NUM_BATCH_BEFORE_PRINT = 10

# model (checkpoint) file params
DATETIME_FORMAT = '%Y%m%d_%H%M'
MODELS_FOLDER = 'models'
MODELS_PREFIX = '{0}_{1}_{2}_{3}.mdl'

# log files
TRN_LOG_FILE = 'training.log'
TST_LOG_FILE = 'testing.log'
LOG_DATETIME_FORMAT = '%Y-%m-%d %H:%M:%S.%f'
