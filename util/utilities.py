# -*- coding:utf-8 -*-
"""
Utilities

Collection of utility functions
"""

import tensorflow as tf
import glob
import os
import random
import common.constants as const
import pathlib
import numpy as np
import json
from datetime import datetime

def get_randomized_image_list(data_dir):
    """
    Gets list of images and labels in data_dir and shuffles them.
    This function expects the directory structure to be:

        data_dir/<label0>/image1.png
        data_dir/<label0>/image2.png
        ...
        data_dir/<labelN>/imageABC.png

    Aside from shuffled lists of images and labels, this function
    also returns a list of the unique labels (classes).
    """

    filenames = []
    labels = []

    data_dir = data_dir.strip('./')
    # get list of images
    filenames = glob.glob(os.path.join(data_dir, '*/*' + const.IMG_EXT))

    # Construct the list of image files and labels
    for filename in filenames:
        labels.append(os.path.basename(os.path.dirname(filename)))

    # Randomize sequence of images
    random_index = list(range(len(filenames)))
    random.seed()
    random.shuffle(random_index)

    filenames = [filenames[idx] for idx in random_index]
    labels = [labels[idx] for idx in random_index]
    
    # return a sorted list of distinct labels
    # sorting is important to achieve consistency 
    # of intenger->class for training/testing set
    classes = sorted(list(set(labels)))

    return filenames, labels, classes

def map_labels_to_classes(labels, classes):
    """
    Replaces string labels with integers and creates
    a dictionary that maps the string-to-integer relationship
    """
    
    nd_labels = np.array(labels)
    class_map = {}

    for int_label in range(len(classes)):
        # get all index where element is equal to class
        class_idxs = np.where(nd_labels == classes[int_label])[0]
        nd_labels[class_idxs] = int_label

        class_map[int_label] = classes[int_label]

    labels = nd_labels.tolist()
    return labels, class_map

def convert_to_feature(value, feature_type):
    """
    Convert input value to TF Feature based on given Feature type
    """

    feature = None
    value = [value]

    if feature_type == 'int64':
        feature = \
            tf.train.Feature(int64_list=tf.train.Int64List(value=value))

    elif feature_type == 'bytes':
        feature = \
            tf.train.Feature(bytes_list=tf.train.BytesList(value=value))

    else:
        error_msg = "Feature not supported!"
        raise Exception(error_msg)

    return feature

def dump_metadata(class_map, num_data, tf_rec_name):
    """
    Dump metadata (number of data, number of classes, intenger->class mapping)
    into a JSON file with same name as associated TFRecords file
    """

    if not os.path.isdir(const.RECORDS_DIR):
        os.mkdir(const.RECORDS_DIR)

    fn = tf_rec_name.replace(const.TF_REC_EXT, '.json')
    fn = os.path.join(const.RECORDS_DIR, fn)

    metadata = {
        'num_data': num_data,
        'num_classes': len(class_map),
        'class_map': class_map
    }

    with open(fn, 'w') as json_file:
        json.dump(metadata, json_file)

def save_as_TFRecord(filenames, labels, tf_rec_name, shape):
    """
    Save images and corresponding labels as TFRecords

    Conversion flow:
        Data -> FeatureSet -> Example -> Serialized Example -> TF Record
    """

    if not os.path.isdir(const.RECORDS_DIR):
        os.mkdir(const.RECORDS_DIR)

    writer = tf.python_io.TFRecordWriter(os.path.join(const.RECORDS_DIR, tf_rec_name))

    # convert features to tf.train.Feature
    for idx in range(len(filenames)):

        # Read image as bytes
        with tf.gfile.FastGFile(filenames[idx], 'rb') as file:
            image_data = file.read()

        # convert features/labels to Feature, then to Examples
        example = tf.train.Example(features=tf.train.Features(
            feature={
                'label': convert_to_feature(int(labels[idx]), 'int64'),
                'pixels': convert_to_feature(image_data, 'bytes'),
            }))

        # Serialize Example, then write as TFRecord
        writer.write(example.SerializeToString())

    writer.close()

def read_metadata(json_path):
    """
    Read number of data and number of classes from JSON file
    """

    with open(json_path) as json_file:
        metadata = json.load(json_file)

    return metadata['num_data'], metadata['num_classes']

def example_parser(serialized_example):
    """
    Parses an Example in a TFRecord.
    Used by Dataset.map()
    """

    # define Features of Example to parse
    feature_set = {
        'label': tf.FixedLenFeature([], dtype=tf.int64),
        'pixels': tf.FixedLenFeature([], dtype=tf.string),
    }

    # parse Example
    features = tf.parse_single_example(serialized_example, features=feature_set)

    image = tf.image.decode_image(features['pixels'])
    image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    image = tf.image.per_image_standardization(image)

    label = features['label']

    return image, label

def tfg_read_from_TFRecord(records_file, buf_size, batch_size):
    """
    Converts data from TFRecords file into a Dataset,
    and creates an Iterator for consuming a batch of the Dataset
    """

    # create dataset from TFRecords
    dataset = tf.data.TFRecordDataset(records_file)

    # parse record into tensors
    dataset = dataset.map(example_parser)

    # shuffle dataset
    dataset = dataset.shuffle(buffer_size=buf_size)

    # repeat occurence of inputs
    dataset = dataset.repeat()

    # generate batches
    dataset = dataset.batch(batch_size)

    # create one-shot iterator
    it = dataset.make_one_shot_iterator()

    # get features
    image, label = it.get_next()

    return image, label

def save_model(step, basename, alias, tf_saver, sess):
    """
    Saves current values of trained model
    """

    if not os.path.isdir(const.MODELS_FOLDER):
        os.mkdir(const.MODELS_FOLDER)

    checkpoint_path = os.path.join(const.MODELS_FOLDER, basename)
    if not os.path.isdir(checkpoint_path):
        os.mkdir(checkpoint_path)

    checkpoint_path = os.path.join(checkpoint_path, const.MODELS_PREFIX)

    # filename format arrangement: date, arch, alias, epoch
    checkpoint_file = checkpoint_path.format(
        datetime.today().strftime(const.DATETIME_FORMAT),
        basename,
        alias,
        step
    )

    save_path = tf_saver.save(sess, checkpoint_file)
    print("Model saved in path: %s" % save_path)

def open_log_file(arch, alias, mode):

    if mode == 'trn':
        base_fn = const.TRN_LOG_FILE

    elif mode == 'tst':
        base_fn = const.TST_LOG_FILE

    log_path = os.path.join(const.MODELS_FOLDER, arch,
        "{0}_{1}_{2}_{3}".format(
            datetime.today().strftime(const.DATETIME_FORMAT),
            arch,
            alias,
            base_fn
            ))

    create_parent_dirs_if_not_exist(log_path)
    log_file = open(log_path, 'w')
    return log_file

def print_logs(file, str):
    """
    Print to file and terminal
    """
    file.write(str + '\n')
    print(str)

def create_parent_dirs_if_not_exist(path):
    pathlib.Path(os.path.dirname(path)).mkdir(parents=True, exist_ok=True)