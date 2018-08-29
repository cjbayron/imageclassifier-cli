# -*- coding:utf-8 -*-
"""
Neural Network Architectures

This contains clasess of different Neural Network architectures.
For each class, the actual network is defined by tfg_network().
"""

import tensorflow as tf
import numpy as np
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

'''
SimpleANN: Simple Fully Connected Network
'''
class SimpleANN():

    def __init__(self, mode, num_labels):
        self.__mode = mode
        self.__num_labels = num_labels

    def tfg_network(self, features):

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