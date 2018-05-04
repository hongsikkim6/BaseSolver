import tensorflow as tf
import sys
import os

from net.network import Network


class VGG16(Network):
    def __init__(self, num_classes):
        Network.__init__(self)
        self._losses = {}
        self._predictions = {}
        self._num_classes = num_classes
        self._num_batch = 256

    def create_architecture(self, is_training):
        self._img_batch = tf.placeholder(dtype=tf.float32,
                                         shape=[None, 32, 32, 3],
                                         name='input')
        self._label = tf.placeholder(dtype=tf.int32, shape=[None],
                                          name='labels')

        regularizer = tf.contrib.layers.l2_regularizer(0.9)
        biases_regularizer = tf.no_regularizer

        initializer = tf.random_normal_initializer(mean=0.0, stddev=0.01)
        with tf.variable_scope('vgg_16') as scope:
            net = self.add_block(
                inputs=self._img_batch,
                repeat=2,
                is_training=is_training,
                filters=64,
                scope_name='block1',
                rate=0.3
            )
            net = self.add_block(
                inputs=net,
                repeat=2,
                is_training=is_training,
                filters=128,
                scope_name='block2',
                rate=0.4
            )
            net = self.add_block(
                inputs=net,
                repeat=3,
                is_training=is_training,
                filters=256,
                scope_name='block3',
                rate=0.4
            )
            net = self.add_block(
                inputs=net,
                repeat=3,
                is_training=is_training,
                filters=512,
                scope_name='block4',
                rate=0.4
            )
            net = self.add_block(
                inputs=net,
                repeat=3,
                is_training=is_training,
                filters=512,
                scope_name='block5',
                rate=0.4
            )

            net = tf.layers.dropout(
                inputs=net,
                rate=0.5
            )
            net = tf.layers.flatten(
                inputs=net
            )
            net = tf.layers.dense(
                inputs=net,
                units=512,
                activation=tf.nn.relu,
                name='dense'
            )
            net = tf.layers.batch_normalization(
                inputs=net,
                trainable=is_training
            )
            net = tf.layers.dropout(
                inputs=net,
                rate=0.5
            )
            cls_score = tf.layers.dense(
                inputs=net,
                units=self._num_classes,
                activation=None
            )
        self._predictions['score'] = cls_score

        self.add_loss()
        print('Done')

        return self._losses

    def add_loss(self):
        with tf.variable_scope('LOSS') as scope:
            cross_entropy = tf.reduce_mean(
                tf.nn.sparse_softmax_cross_entropy_with_logits(
                    logits=self._predictions['score'],
                    labels=self._label
                )
            )
            """
            regularization_loss = tf.add_n(
                tf.losses.get_regularization_losses(), 'regu'
            )
            """
            self._losses['cross_entropy'] = cross_entropy
            self._losses['total_loss'] = cross_entropy

    def train_step(self, sess, data, train_op):
        raise NotImplementedError

    def train_step_with_summary(self, sess, data, train_op):
        raise NotImplementedError

    def get_summary(self, val_data):
        raise NotImplementedError

    def add_block(self, inputs, repeat,
                  is_training, filters, scope_name, rate):
        with tf.variable_scope(scope_name) as scope:
            net = tf.layers.conv2d(
                inputs=inputs,
                filters=filters,
                kernel_size=[3, 3],
                padding='same',
                activation=tf.nn.relu,
                name='conv1'
            )
            net = tf.layers.batch_normalization(
                inputs=net,
                trainable=is_training,
                name='bn1'
            )
            for i in range(repeat - 1):
                net = tf.layers.dropout(
                    inputs=net,
                    rate=rate,
                    name='dropout_' + str(i)
                )
                net = tf.layers.conv2d(
                    inputs=net,
                    filters=filters,
                    kernel_size=[3, 3],
                    padding='same',
                    activation=tf.nn.relu,
                    name='conv_' + str(i)
                )
                net = tf.layers.batch_normalization(
                    inputs=net,
                    trainable=is_training,
                    name='bn_' + str(i)
                )

            net = tf.layers.max_pooling2d(
                inputs=net,
                pool_size=(2, 2),
                strides=(2, 2),
                name='pool'
            )

        return net


if __name__ == '__main__':
    PATH = '/home/hongsikkim/HDD/data/CIFAR-100/'
    net = VGG16(num_classes=5)
    net.create_architecture(is_training=True)

    print('Hello')