# -*- coding:utf-8 -*-
"""
Image Classifier

This contains the ImageClassifier class.
"""

import tensorflow as tf
import os
import math
import numpy as np

import util.utilities as ut
import arch.arch as arch
import common.constants as const

model_arch = {
    'TFCNN' : arch.TFCNN,
    'SimpleANN' : arch.SimpleANN,
    'ImageLSTM' : arch.ImageRNN,
    'ImageGRU' : arch.ImageRNN,   
}

class ImageClassifier():

    def __init__(self, arch, mode, num_classes):
        # instantiate Model Architecture

        if model_arch[arch].__name__ == 'ImageRNN':
            self.__arch = model_arch[arch](mode, num_classes, arch)
        else:
            self.__arch = model_arch[arch](mode, num_classes)

        self.arch_name = arch

    def train(self, alias, records_file, num_data):

        """
        TRAINING GRAPH: START
        """

        # read from TFRecords
        image, label = ut.tfg_read_from_TFRecord(records_file, num_data,
            const.BATCH_SIZE)

        # feed image to network
        logits = self.__arch.tfg_network(image)

        # get loss
        loss = tf.losses.sparse_softmax_cross_entropy(labels=label, logits=logits)

        # optimize based on loss
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=const.LRN_RATE)
        update_model = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

        """
        TRAINING GRAPH: END
        """

        init = tf.global_variables_initializer() # IMPORTANT
        saver  = tf.train.Saver()

        # initialize log file
        log_file = ut.open_log_file(self.arch_name, alias, 'trn')

        processed_num = 0
        epoch_num = 1
        num_steps = math.ceil((num_data * const.NUM_EPOCHS) / const.BATCH_SIZE)

        with tf.Session() as sess:
            
            sess.run(init)

            print("Training on %d images for %d epochs..." 
                % (num_data, const.NUM_EPOCHS))

            # 1 step is 1 batch
            for step in range(num_steps):

                _, loss_val = sess.run([update_model, loss])
                if step % const.NUM_BATCH_BEFORE_PRINT_LOSS == 0:
                    ut.print_logs(log_file, "Epoch: %d/%d, Processed: %d/%d,"
                        " Loss: %f" % (epoch_num, const.NUM_EPOCHS,
                                       processed_num, num_data, loss_val))

                # update counters and save model
                processed_num += const.BATCH_SIZE
                if processed_num >= num_data:
                    processed_num = processed_num % const.BATCH_SIZE
                    if epoch_num % const.NUM_EPOCH_BEFORE_CHKPT == 0 \
                        and epoch_num != const.NUM_EPOCHS:
                        ut.save_model(epoch_num, self.arch_name,
                            alias, saver, sess)

                    epoch_num += 1

            # save last model
            ut.save_model(epoch_num - 1, self.arch_name,
                alias, saver, sess)
        
        log_file.close()

    def classify(self, records_file, checkpoint_path, alias, num_data):

        """
        TESTING GRAPH: START
        """

        # read from TFRecords
        image, labels = ut.tfg_read_from_TFRecord(records_file, num_data,
            const.TST_BATCH_SIZE)

        # feed image to network
        logits = self.__arch.tfg_network(image)
        softmax = tf.nn.softmax(logits=logits)
        pred = tf.argmax(softmax, axis=1)

        """
        TESTING GRAPH: END
        """

        init = tf.global_variables_initializer() # IMPORTANT
        saver  = tf.train.Saver()

        # initialize log file
        log_file = ut.open_log_file(self.arch_name, alias, 'tst')

        num_steps = math.ceil(num_data / const.TST_BATCH_SIZE)
        correct = 0

        with tf.Session() as sess:
            
            sess.run(init)

            ut.print_logs(log_file,
                "Starting classification. Using model in:\n%s" % (checkpoint_path))
            saver.restore(sess, checkpoint_path)

            for step in range(num_steps):

                # get prediction and labels for this iteration
                step_pred, step_labels = sess.run([pred, labels])
                
                # compare prediction and label
                correct += np.sum(step_pred == step_labels)

                if step % const.NUM_BATCH_BEFORE_PRINT == 0:
                    ut.print_logs(log_file, 
                        "Processed %d/%d images" % ((step + 1)*const.TST_BATCH_SIZE,
                        num_data))

            acc = (correct / num_data) * 100
            ut.print_logs(log_file, "Accuracy: %f" % (acc))

        log_file.close()