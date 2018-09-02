# -*- coding:utf-8 -*-
"""
Neural Network Architectures

This contains clasess of different Neural Network architectures.
For each class, the actual network is defined by tfg_network().
"""

import numpy as np
import tensorflow as tf
import common.constants as const

class TFCNN():
    """
    TFCNN: Tensorflow CNN Architecture (CNN MNIST Classifier)
    Source: https://www.tensorflow.org/tutorials/estimators/cnn
    """

    def __init__(self, mode, num_labels):
        self.__mode = mode
        self.__num_labels = num_labels

    def tfg_network(self, features):
        """
        Neural Network Architecture for TFCNN
        """

        input_dim = [-1] # for batch size
        input_dim.extend(const.IMG_SHAPE)

        # Input Layer
        input_layer = tf.reshape(features, input_dim)

        # Convolutional Layer #1
        conv1 = tf.layers.conv2d(
            inputs=input_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #1
        pool1 = tf.layers.max_pooling2d(
            inputs=conv1,
            pool_size=[2, 2],
            strides=2)

        # Convolutional Layer #2
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu)

        # Pooling Layer #2
        pool2 = tf.layers.max_pooling2d(
            inputs=conv2,
            pool_size=[2, 2],
            strides=2)

        # Dense Layer
        pool2_flat = tf.layers.Flatten()(pool2)
        dense = tf.layers.dense(
            inputs=pool2_flat,
            units=1024,
            activation=tf.nn.relu)

        # Dropout
        dropout = tf.layers.dropout(
            inputs=dense,
            rate=0.4,
            training=(self.__mode == const.TRN_MODE))

        # Logits Layer
        logits = tf.layers.dense(
            inputs=dropout,
            units=self.__num_labels)

        return logits


class SimpleANN():
    """
    SimpleANN: Simple Fully Connected Network
    """

    def __init__(self, mode, num_labels):
        self.__mode = mode
        self.__num_labels = num_labels

    def tfg_network(self, features):
        """
        Neural Network Architecture for SimpleANN
        """

        # Input Layer (Flattened)
        input_layer = tf.reshape(features, [-1, np.prod(const.IMG_SHAPE)])

        # Hidden Layer (Dense)
        hidden_1 = tf.layers.dense(
            inputs=input_layer,
            units=1024,
            activation=tf.nn.relu)

        # Logits Layer
        logits = tf.layers.dense(
            inputs=hidden_1,
            units=self.__num_labels)

        return logits


class ImageRNN():
    """
    ImageRNN: RNN for Image Classification

    The idea here is to treat each row of an image as data for a single time step,
    and the number of columns i.e. height of image is treated as no. of sequences.

    In this class we build a many-to-one RNN architecture, where the input is
    a sequence of single-row pixel values, and the output is an image label.
    """

    def __init__(self, mode, num_labels, cell):
        self.__mode = mode
        self.__num_labels = num_labels
        self.__cell = cell

    def tfg_network(self, features):
        """
        Neural Network Architecture for ImageRNN (ImageLSTM/ImageGRU)
        """

        num_steps = const.IMG_SHAPE[0]

        img_wd = const.IMG_SHAPE[1]
        img_ch = const.IMG_SHAPE[2]

        img_shape = [-1] # batch size
        img_shape.extend(const.IMG_SHAPE)

        # this reshapes image to batch_size x H x W x C
        image_batch = tf.reshape(features, img_shape)
        # now flatten data in W, C dimensions
        image_batch = tf.reshape(image_batch, [-1, num_steps, img_wd * img_ch])

        # Input Layer
        # unstack before passing to static_rnn
        input_layer = tf.unstack(image_batch, axis=1)

        # RNN cell
        # output_size == num_units
        cell = None
        if self.__cell == 'ImageLSTM':
            cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=512)

        elif self.__cell == 'ImageGRU':
            cell = tf.nn.rnn_cell.GRUCell(num_units=512)

        # we use a static RNN here since we expect data in
        # each step of our sequence to be of same size
        outputs, state = tf.nn.static_rnn(cell, input_layer, dtype=tf.float32)

        # for Static RNN vs Dynamic RNN see:
        # https://stackoverflow.com/questions/43100981/what-is-a-dynamic-rnn-in-tensorflow

        # "outputs" is a list of outputs at each time_step, so we only pick the last
        final_out = outputs[-1]

        # Logits Layer
        logits = tf.layers.dense(
            inputs=final_out,
            units=self.__num_labels)

        return logits
