"""Biological feature calculation utilities

This module contains code for calculating biological features using 
the raw sequence of a given prime editor guide RNA (pegRNA) sequence.

If the function name does not specify RNA/DNA, then both sequence
types are supported
"""
import os, sys

import pandas as pd
import numpy as np
# Calculating Minimum Free Energy (MFE)
import RNA 
# Calculating melting temperature
from Bio.Seq import Seq
from Bio.SeqUtils import MeltingTemp as mt 
import tensorflow as tf

from .sequence_utils import onehot_encode
from .constants import DEEPSPCAS9_MODEL_DIR, DEVICE

def calculate_rna_mfe(sequence: str) -> float:
    """
    Calculate the Minimum Free Energy (MFE) of a given RNA sequence
    using ViennaRNA package.
    
    Args:
        sequence: RNA sequence string
        
    Returns:
        Minimum free energy value in kcal/mol
    """
    # Use ViennaRNA's fold function to calculate MFE
    structure, mfe = RNA.fold(sequence)
    return mfe


def calculate_mfe(sequence: str) -> float:
    """
    Backwards-compatible alias for calculate_rna_mfe.

    Args:
        sequence: RNA sequence string

    Returns:
        Minimum free energy value in kcal/mol
    """
    return calculate_rna_mfe(sequence)


def calculate_mt_wallace(sequence: str) -> float:
    """
    Calculate the melting temperature of a given sequence using
    the Wallace method.
    
    Args:
        sequence: DNA/RNA sequence string
        
    Returns:
        Melting temperature in degrees Celsius
    """
    # Convert the sequence to a Bio.Seq object
    seq = Seq(sequence)
    # Calculate the melting temperature
    tm = mt.Tm_Wallace(seq)
    return tm


def calculate_gc_content(sequence: str) -> float:
    """
    Calculate the GC content of a DNA/RNA sequence.
    
    Args:
        sequence: DNA/RNA sequence string
        
    Returns:
        GC content as a fraction (0.0 to 1.0)
    """
    sequence = sequence.upper()
    gc_count = sequence.count('G') + sequence.count('C')
    total_count = len(sequence)
    
    if total_count == 0:
        return 0.0
    
    return gc_count / total_count


# === DeepSpCas9 Score Calculation from DeepPrime ===
class Deep_SpCas9(object):
    def __init__(self, filter_size, filter_num, node_1=80, node_2=60, l_rate=0.005):
        length = 30
        self.inputs      = tf.compat.v1.placeholder(tf.float32, [None, 1, length, 4])
        self.targets     = tf.compat.v1.placeholder(tf.float32, [None, 1])
        self.is_training = tf.compat.v1.placeholder(tf.bool)

        def create_new_conv_layer(input_data, num_input_channels, num_filters, filter_shape, pool_shape, name):
            # setup the filter input shape for tf.compat.v1.nn.conv_2d
            conv_filt_shape = [filter_shape[0], filter_shape[1], num_input_channels,
                               num_filters]

            # initialise weights and bias for the filter
            w = tf.compat.v1.Variable(tf.compat.v1.truncated_normal(conv_filt_shape, stddev=0.03), name=name + '_W')
            b = tf.compat.v1.Variable(tf.compat.v1.truncated_normal([num_filters]), name=name + '_b')

            # setup the convolutional layer operation
            out_layer = tf.nn.conv2d(input_data, w, [1, 1, 1, 1], padding='VALID')

            # add the bias
            out_layer += b

            # apply a ReLU non-linear activation
            out_layer = tf.python.keras.layers.Dropout(rate=0.3)(tf.nn.relu(out_layer))

            # now perform max pooling
            ksize     = [1, pool_shape[0], pool_shape[1], 1]
            strides   = [1, 1, 2, 1]
            out_layer = tf.nn.avg_pool(out_layer, ksize=ksize, strides=strides, padding='SAME')

            return out_layer

        # def end: create_new_conv_layer

        L_pool_0 = create_new_conv_layer(self.inputs, 4, filter_num[0], [1, filter_size[0]], [1, 2], name='conv1')
        L_pool_1 = create_new_conv_layer(self.inputs, 4, filter_num[1], [1, filter_size[1]], [1, 2], name='conv2')
        L_pool_2 = create_new_conv_layer(self.inputs, 4, filter_num[2], [1, filter_size[2]], [1, 2], name='conv3')

        with tf.compat.v1.variable_scope('Fully_Connected_Layer1'):
            layer_node_0 = int((length - filter_size[0]) / 2) + 1
            node_num_0   = layer_node_0 * filter_num[0]
            layer_node_1 = int((length - filter_size[1]) / 2) + 1
            node_num_1   = layer_node_1 * filter_num[1]
            layer_node_2 = int((length - filter_size[2]) / 2) + 1
            node_num_2   = layer_node_2 * filter_num[2]

            L_flatten_0  = tf.reshape(L_pool_0, [-1, node_num_0])
            L_flatten_1  = tf.reshape(L_pool_1, [-1, node_num_1])
            L_flatten_2  = tf.reshape(L_pool_2, [-1, node_num_2])
            L_flatten    = tf.concat([L_flatten_0, L_flatten_1, L_flatten_2], 1, name='concat')

            node_num     = node_num_0 + node_num_1 + node_num_2
            W_fcl1       = tf.compat.v1.get_variable("W_fcl1", shape=[node_num, node_1])
            B_fcl1       = tf.compat.v1.get_variable("B_fcl1", shape=[node_1])
            L_fcl1_pre   = tf.nn.bias_add(tf.matmul(L_flatten, W_fcl1), B_fcl1)
            L_fcl1       = tf.nn.relu(L_fcl1_pre)
            L_fcl1_drop  = tf.python.keras.layers.Dropout(rate=0.3)(L_fcl1)

        with tf.compat.v1.variable_scope('Fully_Connected_Layer2'):
            W_fcl2       = tf.compat.v1.get_variable("W_fcl2", shape=[node_1, node_2])
            B_fcl2       = tf.compat.v1.get_variable("B_fcl2", shape=[node_2])
            L_fcl2_pre   = tf.nn.bias_add(tf.matmul(L_fcl1_drop, W_fcl2), B_fcl2)
            L_fcl2       = tf.nn.relu(L_fcl2_pre)
            L_fcl2_drop  = tf.python.keras.layers.Dropout(rate=0.3)(L_fcl2)

        with tf.compat.v1.variable_scope('Output_Layer'):
            W_out        = tf.compat.v1.get_variable("W_out", shape=[node_2, 1])
            B_out        = tf.compat.v1.get_variable("B_out", shape=[1])
            self.outputs = tf.nn.bias_add(tf.matmul(L_fcl2_drop, W_out), B_out)

        # Define loss function and optimizer
        self.obj_loss    = tf.reduce_mean(tf.square(self.targets - self.outputs))
        self.optimizer   = tf.compat.v1.train.AdamOptimizer(l_rate).minimize(self.obj_loss)

# class end: Deep_xCas9
def Model_Finaltest(sess, TEST_X, model):
    test_batch = 500
    test_spearman = 0.0
    optimizer = model.optimizer
    TEST_Z = np.zeros((TEST_X.shape[0], 1), dtype=float)

    for i in range(int(np.ceil(float(TEST_X.shape[0]) / float(test_batch)))):
        Dict = {model.inputs: TEST_X[i * test_batch:(i + 1) * test_batch], model.is_training: False}
        TEST_Z[i * test_batch:(i + 1) * test_batch] = sess.run([model.outputs], feed_dict=Dict)[0]

    list_score = sum(TEST_Z.tolist(), [])

    return list_score

def calculate_DeepSpCas9_score(list_target30):
    """Calculate DeepSpCas9 scores for a list of 30-nt target sequences.
    [-4 bp] + [20-nt guide] + [NGG PAM] + [+3 bp]

    Args:
        list_target30: List of 30-nt target sequences.
    Returns:
        List of DeepSpCas9 scores.
    """
    # TensorFlow config
    conf = tf.compat.v1.ConfigProto()
    # ask TensorFlow to use available device

    best_model_cv = 0.0

    TEST_X = np.zeros((len(list_target30), 1, 30, 4), dtype=float)
    for i in range(len(list_target30)):
        TEST_X[i, 0, :, :] = onehot_encode(list_target30[i])
    TEST_X_nohot = list_target30

    best_model = 'PreTrain-Final-3-5-7-100-70-40-0.001-550-80-60'
    valuelist = best_model.split('-')
    fulllist = []

    for value in valuelist:
        if value == 'True':
            value = True
        elif value == 'False':
            value = False
        else:
            try:
                value = int(value)
            except:
                try:
                    value = float(value)
                except:
                    pass
        fulllist.append(value)

    filter_size_1, filter_size_2, filter_size_3, filter_num_1, filter_num_2, filter_num_3, l_rate, load_episode, node_1, node_2 = fulllist[
                                                                                                                                  2:]
    filter_size = [filter_size_1, filter_size_2, filter_size_3]
    filter_num = [filter_num_1, filter_num_2, filter_num_3]
    if3d = False
    inception = False
    args = [filter_size, filter_num, l_rate, load_episode]
    tf.compat.v1.reset_default_graph()
    with tf.compat.v1.Session(config=conf) as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        model = Deep_SpCas9(filter_size, filter_num, node_1, node_2, args[2])

        saver = tf.compat.v1.train.Saver()
        saver.restore(sess, DEEPSPCAS9_MODEL_DIR.joinpath(best_model + '.ckpt'))
        list_score = Model_Finaltest(sess, TEST_X, model)

    return list_score
