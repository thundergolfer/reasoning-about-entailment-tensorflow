# coding: utf-8
import tensorflow as tf
import numpy as np

from networks.lstm import LSTMCell


class AttentionLSTMCell(LSTMCell):
    def __init__(self, num_units, hiddens, states, **kwargs):
        super(AttentionLSTMCell, self).__init__(num_units=num_units)

        # warm-up
        self.warm_hiddens = hiddens
        self.L = len(self.warm_hiddens)
        self.Y = tf.expand_dims(tf.transpose(tf.pack(self.warm_hiddens), [2, 1, 0]), 3)
        self.c = [states[-1]]

        # weights
        self.w_y = self.get_4Dweights(filter_height=self._num_units, filter_width=1, in_channels=1, out_channels=self._num_units, name="w_y")
        self.w = self.get_4Dweights(filter_height=self._num_units, filter_width=1, in_channels=1, out_channels=1, name="w")
        self.w_h = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_h")
        self.w_r = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_r")
        self.w_t = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_t")
        self.w_p = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_p")
        self.w_x = self.get_weights(dim_in=self._num_units, dim_out=self._num_units, name="w_x")

    def initialize_something(self, input):
        super(AttentionLSTMCell, self).initialize_something(input=input)

        # warm-up
        self.h = [self.warm_hiddens[-1]]

        # attention
        self.r = [self.get_biases(dim_out=self._num_units, name="r", trainable=False) * self.batch_size_vector]

    def process(self, input, delimiter=True):
        # classic-LSTM module
        super(AttentionLSTMCell, self).process(input=input)

        # attention-LSTM module
        if not delimiter:
            first_term = tf.transpose(tf.nn.conv2d(input=self.Y, filter=self.w_y, strides=[1, 1, 1, 1], padding="VALID"), [0, 3, 2, 1])
            second_term = tf.expand_dims(tf.transpose(tf.tile(tf.expand_dims(tf.matmul(self.w_h, self.h[-1]) + tf.matmul(self.w_r, self.r[-1]), [2]), [1, 1, self.L]), [1, 0, 2]), 3)
            M = tf.tanh(first_term + second_term)
            alpha = tf.expand_dims(tf.nn.softmax(tf.squeeze(tf.nn.conv2d(input=M, filter=self.w, strides=[1, 1, 1, 1], padding="VALID"), [1, 3])), 2)
            r = tf.transpose(tf.squeeze(tf.batch_matmul(tf.squeeze(self.Y, [3]), alpha), [2]), [1, 0]) + tf.tanh(tf.matmul(self.w_t, self.r[-1]))
            self.r.append(r)

    @property
    def features(self):
        return tf.tanh(tf.matmul(self.w_p, self.r[-1]) + tf.matmul(self.w_x, self.h[-1]))
