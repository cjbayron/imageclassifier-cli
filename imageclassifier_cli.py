# -*- coding:utf-8 -*-
"""
Image Classifier CLI
"""

import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import glob

import util.utilities as ut
import common.constants as const
from image_classifier import MODEL_ARCH, ImageClassifier

if __name__ == '__main__':
    # Command line arguments
    ARG_PARSER = ArgumentParser(formatter_class=RawTextHelpFormatter)
    ARG_PARSER.add_argument('-m', dest='mode',
                            required=True,
                            choices=[const.TRN_PREP_MODE,
                                     const.TST_PREP_MODE,
                                     const.TRN_MODE,
                                     const.TST_MODE],
                            help='Execution mode:\n'
                            + '    ' + const.TRN_PREP_MODE
                            + ' - prepare data as TFRecords then execute training\n'
                            + '    ' + const.TST_PREP_MODE
                            + ' - prepare data as TFRecords then execute testing\n'
                            + '    ' + const.TRN_MODE
                            + ' - Execute training (no preparation)\n'
                            + '    ' + const.TST_MODE
                            + ' - Execute testing (no preparation)\n')
    ARG_PARSER.add_argument('-d', dest='data_dir',
                            required=False,
                            help='Location of images to use\n - used for '
                            + const.TRN_PREP_MODE + ' and ' + const.TST_PREP_MODE)
    ARG_PARSER.add_argument('-r', dest='tf_rec',
                            required=True,
                            help='TFRecords file\n - NAME of TFRecords file '
                            + 'to be created (for trn_prep, tst_prep)\n'
                            + ' - PATH of TFRecords file to be used (for '
                            + const.TRN_MODE + ', tst)')
    ARG_PARSER.add_argument('-a', dest='arch',
                            required=True,
                            choices=sorted([key for key in MODEL_ARCH]),
                            help='Model Architecture to Use')
    ARG_PARSER.add_argument('-l', dest='alias',
                            required=True,
                            help='Alias for trained model (e.g. name of data)')
    ARG_PARSER.add_argument('-s', dest='model_dir',
                            required=False,
                            help='Location of Saved Model\n - optional; used only for '
                            + const.TST_MODE + ' and ' + const.TST_PREP_MODE)
    ARG_PARSER.add_argument('-e', dest='model_epoch',
                            required=False,
                            help='Epoch (load model saved at end of specific epoch)\n'
                            + ' - optional; used only for '
                            + const.TST_MODE + ' and ' + const.TST_PREP_MODE)


    ARGS = ARG_PARSER.parse_args()

    # this will store the detected number of classes (unique labels)
    NUM_CLASSES = 0
    # this will store number of data to use for training / testing
    NUM_DATA = 0

    if 'prep' in ARGS.mode:
        if ARGS.data_dir is None:
            ARG_PARSER.error("No specified data to prepare. Use -d argument.")

        if not os.path.isdir(ARGS.data_dir):
            ARG_PARSER.error("Dataset directory does not exist!")

        # get randomized list of filenames, labels, and classes
        FILE_NAMES, LABELS, CLASSES = ut.get_randomized_image_list(ARGS.data_dir)
        # replace labels with integer values, and get corresponding mapping
        LABELS, CLASS_MAP = ut.map_labels_to_classes(LABELS, CLASSES)

        NUM_DATA = len(FILE_NAMES)
        NUM_CLASSES = len(CLASSES)

        print("Found %d images. Creating TF records... " % (NUM_DATA))

        # fix filename
        TF_REC = ARGS.tf_rec
        if TF_REC.find(const.TF_REC_EXT) == -1:
            TF_REC += const.TF_REC_EXT

        # save metadata
        ut.dump_metadata(CLASS_MAP, NUM_DATA, TF_REC)
        # save images to TFRcords
        ut.save_as_TFRecord(FILE_NAMES, LABELS, TF_REC)

        print("Done.")
        TF_REC = os.path.join(const.RECORDS_DIR, TF_REC)

    else:
        if not os.path.exists(ARGS.tf_rec):
            ARG_PARSER.error("TF Records file does not exist!")

        # check if metadata file (JSON) exists
        JSON_PATH = ARGS.tf_rec.replace(const.TF_REC_EXT, '.json')
        if not os.path.exists(JSON_PATH):
            ARG_PARSER.error("Metadata (JSON) of TF Records file does not exist!")

        TF_REC = ARGS.tf_rec

        # get NUM_DATA, NUM_CLASSES from metadata
        NUM_DATA, NUM_CLASSES = ut.read_metadata(JSON_PATH)


    # instantiate model
    IMG_MODEL = ImageClassifier(ARGS.arch, ARGS.mode, NUM_CLASSES)

    # perform training
    if const.TRN_MODE in ARGS.mode:
        IMG_MODEL.train(ARGS.alias, TF_REC, NUM_DATA)

    # perform classification
    elif const.TST_MODE in ARGS.mode:

        if ARGS.model_dir is None:
            ARGS.model_dir = os.path.join(const.MODELS_FOLDER, ARGS.arch)

        elif not os.path.isdir(ARGS.model_dir):
            ARG_PARSER.error("Model checkpoint directory does not exist!")

        # wildcards should always follow arrangement in ut.save_model
        # filename format arrangement: date, arch, alias, epoch
        if ARGS.model_epoch is None:
            # get all checkpoint files
            MDL_WILDCARD = os.path.join(ARGS.model_dir, '*_' + ARGS.alias + '*.mdl.data*')

        else:
            MDL_WILDCARD = os.path.join(ARGS.model_dir,
                                        str(ARGS.model_epoch)
                                        + '*_' + ARGS.alias + '*.mdl.data*')

        CHKPT_PATH = glob.glob(MDL_WILDCARD)
        if not CHKPT_PATH:
            ARG_PARSER.error("Checkpoint file does not exist!")

        # purpose of reverse is to get MAX step in case ARGS.model_epoch is not specified
        # sort has no effect if ARGS.model_epoch is specified
        CHKPT_PATH.sort(reverse=True)
        IMG_MODEL.classify(TF_REC, CHKPT_PATH[0].split('.data')[0], ARGS.alias, NUM_DATA)
