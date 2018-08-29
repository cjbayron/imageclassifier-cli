# -*- coding:utf-8 -*-
"""
Image Classifier CLI
"""

import tensorflow as tf
import os
from argparse import ArgumentParser
from argparse import RawTextHelpFormatter
import glob

import util.utilities as ut
import common.constants as const
from image_classifier import *

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
                            choices=[key for key in model_arch],
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
    num_classes = 0
    # this will store number of data to use for training / testing
    num_data = 0

    if 'prep' in ARGS.mode:
        if ARGS.data_dir is None:
            ARG_PARSER.error("No specified data to prepare. Use -d argument.")

        if not os.path.isdir(ARGS.data_dir):
            ARG_PARSER.error("Dataset directory does not exist!")

        # get randomized list of filenames, labels, and classes
        filenames, labels, classes = ut.get_randomized_image_list(ARGS.data_dir)
        # replace labels with integer values, and get corresponding mapping
        labels, class_map = ut.map_labels_to_classes(labels, classes)

        num_data = len(filenames)
        num_classes = len(classes)

        print("Found %d images. Creating TF records... " % (num_data))

        # fix filename
        tf_rec = ARGS.tf_rec
        if tf_rec.find(const.TF_REC_EXT) == -1:
            tf_rec += const.TF_REC_EXT

        # save metadata
        ut.dump_metadata(class_map, num_data, tf_rec)
        # save images to TFRcords
        ut.save_as_TFRecord(filenames, labels, tf_rec, const.IMG_SHAPE)

        print("Done.")
        tf_rec = os.path.join(const.RECORDS_DIR, tf_rec)

    else:
        if not os.path.exists(ARGS.tf_rec):
            ARG_PARSER.error("TF Records file does not exist!")

        # check if metadata file (JSON) exists
        json_path = ARGS.tf_rec.replace(const.TF_REC_EXT, '.json')
        if not os.path.exists(json_path):
            ARG_PARSER.error("Metadata (JSON) of TF Records file does not exist!")

        tf_rec = ARGS.tf_rec

        # get num_data, num_classes from metadata
        num_data, num_classes = ut.read_metadata(json_path)


    # instantiate model
    model = ImageClassifier(ARGS.arch, ARGS.mode, num_classes)

    # perform training
    if const.TRN_MODE in ARGS.mode:
        model.train(ARGS.alias, tf_rec, num_data)

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
            mdl_wildcard = os.path.join(ARGS.model_dir, '*_' + ARGS.alias + '*.mdl.data*')

        else:
            mdl_wildcard = os.path.join(ARGS.model_dir,
                str(ARGS.model_epoch) + '*_' + ARGS.alias + '*.mdl.data*')
        
        checkpoint_path = glob.glob(mdl_wildcard)
        if not checkpoint_path:
            ARG_PARSER.error("Checkpoint file does not exist!")

        # purpose of reverse is to get MAX step in case ARGS.model_epoch is not specified
        # sort has no effect if ARGS.model_epoch is specified
        checkpoint_path.sort(reverse=True)
        model.classify(tf_rec, checkpoint_path[0].split('.data')[0], ARGS.alias, num_data)