# Original code from Jerod Weinman
# https://github.com/weinman/cnn_lstm_ctc_ocr


# CNN-LSTM-CTC-OCR
# Copyright (C) 2017 Jerod Weinman
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>

import tensorflow as tf
from net.subnets.layers import conv_layer, pool_layer

layer_params = [ [  64, 3, 'valid', 'conv1', False],
                 [  64, 3, 'same',  'conv2', True], # pool
                 [ 128, 3, 'same',  'conv3', False],
                 [ 128, 3, 'same',  'conv4', True], # hpool
                 [ 256, 3, 'same',  'conv5', False],
                 [ 256, 3, 'same',  'conv6', True], # hpool
                 [ 512, 3, 'same',  'conv7', False],
                 [ 512, 3, 'same',  'conv8', True]] # hpool 3

rnn_size = 2**9
dropout_rate = 0.5


def Convnet_8(inputs, widths, is_training):
    with tf.variable_scope('convnet') as scope:
        conv1 = conv_layer(inputs, layer_params[0], is_training) # 30,30
        conv2 = conv_layer(conv1, layer_params[1], is_training)  # 30,30
        pool2 = pool_layer(conv2, 2, 'valid', 'pool2')  # 15,15
        conv3 = conv_layer(pool2, layer_params[2], is_training)  # 15,15
        conv4 = conv_layer(conv3, layer_params[3], is_training)  # 15,15
        pool4 = pool_layer(conv4, 1, 'valid', 'pool4')  # 7,14
        conv5 = conv_layer(pool4, layer_params[4], is_training)  # 7,14
        conv6 = conv_layer(conv5, layer_params[5], is_training)  # 7,14
        pool6 = pool_layer(conv6, 1, 'valid', 'pool6')  # 3,13
        conv7 = conv_layer(pool6, layer_params[6], is_training)  # 3,13
        conv8 = conv_layer(conv7, layer_params[7], is_training)  # 3,13
        pool8 = tf.layers.max_pooling2d(conv8, [3, 1], [3, 1],
                                        padding='valid', name='pool8')  # 1,13
        features = tf.squeeze(pool8, axis=1, name='features')  #squeeze row dim
        kernel_sizes = [params[1] for params in layer_params]

        # Calculate resulting sequence length from original image widths
        conv1_trim = tf.constant(2 * (kernel_sizes[0] // 2),
                                 dtype=tf.int32,
                                 name='conv1_trim')
        one = tf.constant(1, dtype=tf.int32, name='one')
        two = tf.constant(2, dtype=tf.int32, name='two')
        after_conv1 = tf.subtract(widths, conv1_trim)
        after_pool2 = tf.floor_div(after_conv1, two)
        after_pool4 = tf.subtract(after_pool2, one)
        after_pool6 = tf.subtract(after_pool4, one)
        after_pool8 = after_pool6

        sequence_length = \
            tf.reshape(after_pool8, [-1], name='seq_len')  # Vectorize

    return features, sequence_length, rnn_size

