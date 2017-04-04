# coding: utf-8
import tensorflow as tf
import numpy as np

from networks.base import TensorflowTrainable


class LSTMCell(TensorflowTrainable):
    def __init__(self, num_units, **kwargs):
        super(LSTMCell, self).__init__()

        # k
        self._num_units = num_units

        # weights
        self.w_i = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_i")
        self.w_f = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_f")
        self.w_o = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_o")
        self.w_c = self.get_weights(dim_in=2 * self._num_units, dim_out=self._num_units, name="w_c")

        # biases
        self.b_i = self.get_biases(dim_out=self._num_units, name="b_i")
        self.b_f = self.get_biases(dim_out=self._num_units, name="b_f")
        self.b_o = self.get_biases(dim_out=self._num_units, name="b_o")
        self.b_c = self.get_biases(dim_out=self._num_units, name="b_c")

        # state
        self.c = [self.get_biases(dim_out=self._num_units, name="c", trainable=False)]

    def initialize_something(self, input):
        self.batch_size_vector = 1 + 0 * tf.expand_dims(tf.unpack(tf.transpose(input, [1, 0]))[0], 0)
        self.h = [self.get_biases(dim_out=self._num_units, name="h", trainable=False) * self.batch_size_vector]

    def process(self, input, **kwargs):
        H = tf.concat(0, [tf.transpose(input, perm=[1, 0]), self.h[-1]])
        i = tf.sigmoid(x=tf.add(tf.matmul(self.w_i, H), self.b_i))
        f = tf.sigmoid(x=tf.add(tf.matmul(self.w_f, H), self.b_f))
        o = tf.sigmoid(x=tf.add(tf.matmul(self.w_o, H), self.b_o))
        c = f * self.c[-1] + i * tf.tanh(x=tf.add(tf.matmul(self.w_c, H), self.b_c))
        h = o * tf.tanh(x=self.c[-1])
        self.c.append(c)
        self.h.append(h)

    @property
    def features(self):
        return self.h[-1]
