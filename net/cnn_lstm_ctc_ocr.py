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
import sys
import os

from net.network import Network
from net.subnets.convnet_8 import Convnet_8
from net.subnets.layers import rnn_layer


class CNN_LSTM_CTC_OCR(Network):
    def __init__(self, num_classes,
                 width=32,
                 convnet=Convnet_8):
        Network.__init__(self)
        self._width = width
        self._losses = {}
        self._predictions = {}
        self._event_summaries = {}
        self._act_summaries = []
        self._score_summaries = {}
        self._train_summaries = []
        self._num_classes = num_classes
        self._convnet = convnet

    def create_architecture(self, is_training=True):
        self._img_batch = tf.placeholder(
            dtype=tf.float32,
            shape=[None, self._width, None, 3],
            name='input'
        )
        self._label = tf.placeholder(
            dtype=tf.int32,
            shape=[None, None],
            name='labels'
        )

        features, sequence_length, rnn_size = self._convnet(
            inputs=self._img_batch,
            widths=self._width,
            is_training=is_training
        )

        # RNN PART
        # Build a stack of RNN layers from input features

        logit_activation = tf.nn.relu
        weight_initializer = tf.contrib.layers.variance_scaling_initializer()
        bias_initializer = tf.constant_initializer(value=0.0)

        with tf.variable_scope('RNN') as scope:
            # Transpose to time-major order for efficiency
            rnn_sequence = tf.transpose(features, perm=[1, 0, 2],
                                        name='time_major')
            rnn1 = rnn_layer(rnn_sequence, sequence_length, rnn_size, 'bdrnn1')
            rnn2 = rnn_layer(rnn1, sequence_length, rnn_size, 'bdrnn2')
            self._predictions['rnn_logits'] = tf.layers.dense(
                inputs=rnn2,
                units=self._num_classes + 1,
                activation=logit_activation,
                kernel_initializer=weight_initializer,
                bias_initializer=bias_initializer,
                name='logits'
            )

        self._losses['ctc_loss'] = tf.nn.ctc_loss(
            labels=self._label,
            inputs=self._predictions['rnn_logits'],
            sequence_length=sequence_length,
            time_major=True
        )
        self._losses['total_loss'] = tf.reduce_mean(self._losses['ctc_loss'])

        self._score_summaries.update(self._predictions)
        self._event_summaries.update(self._losses)
        for var in tf.trainable_variables():
            self._train_summaries.append(var)

        val_summaries = []

        with tf.device('/cpu:0'):
            for key, var in self._event_summaries.items():
                val_summaries.append(tf.summary.scalar(key, var))
            for key, var in self._score_summaries.items():
                self._add_score_summary(key, var)
            for var in self._act_summaries:
                self._add_act_summary(var)
            for var in self._train_summaries:
                self._add_train_summary(var)

        self._summary_op = tf.summary.merge_all()
        self._summary_op_val = tf.summary.merge(val_summaries)

        return self._losses

    def _add_act_summary(self, tensor):
        tf.summary.histogram('ACT/' + tensor.op.name + '/activations', tensor)
        tf.summary.scalar('ACT/' + tensor.op.name + '/zero_fraction',
                          tf.nn.zero_fraction(tensor))

    def _add_score_summary(self, key, tensor):
        tf.summary.histogram('SCORE/' + tensor.op.name + '/' + key + '/scores',
                             tensor)

    def _add_train_summary(self, var):
        tf.summary.histogram('TRAIN/' + var.op.name, var)

    def train_step(self, sess, data, train_op):
        feed_dict = {
            self._img_batch: data['img_batch'],
            self._label: data['label']}
        loss, _ = sess.run(
            [self._losses['total_loss'],
             train_op],
            feed_dict=feed_dict
        )

        return loss

    def train_step_with_summary(self, sess, data, train_op):
        feed_dict = {
            self._img_batch: data['img_batch'],
            self._label: data['label']}
        loss, summary, _ = sess.run(
            [self._losses['total_loss'],
             self._summary_op,
             train_op],
            feed_dict=feed_dict
        )

        return loss, summary

    def get_summary(self, sess, data):
        feed_dict = {
            self._img_batch: data['img_batch'],
            self._label: data['label']}
        summary = sess.run(self._summary_op_val,
                           feed_dict=feed_dict)

        return summary















