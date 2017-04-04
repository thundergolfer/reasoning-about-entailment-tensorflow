# coding: utf-8
import tensorflow as tf
import numpy as np


class TensorflowTrainable(object):
    """
    Add Docstring
    """
    def __init__(self):
        self.parameters = []

    def get_weights(self, dim_in, dim_out, name, trainable=True):
        shape = (dim_out, dim_in)
        weightsInitializer = tf.constant_initializer(self.truncated_normal(shape=shape, stddev=0.01, mean=0.))
        weights = tf.get_variable(initializer=weightsInitializer, shape=shape, trainable=True, name=name)
        if trainable:
            self.parameters.append(weights)
        return weights

    def get_4Dweights(self, filter_height, filter_width, in_channels, out_channels, name, trainable=True):
        shape = (filter_height, filter_width, in_channels, out_channels)
        weightsInitializer = tf.constant_initializer(self.truncated_normal(shape=shape, stddev=0.01, mean=0.))
        weights = tf.get_variable(initializer=weightsInitializer, shape=shape, trainable=True, name=name)
        if trainable:
            self.parameters.append(weights)
        return weights

    def get_biases(self, dim_out, name, trainable=True):
        shape = (dim_out, 1)
        initialBiases = tf.constant_initializer(np.zeros(shape))
        biases = tf.get_variable(initializer=initialBiases, shape=shape, trainable=True, name=name)
        if trainable:
            self.parameters.append(biases)
        return biases

    @staticmethod
    def truncated_normal(shape, stddev, mean=0.):
        rand_init = np.random.normal(loc=mean, scale=stddev, size=shape)
        inf_mask = rand_init < (mean - 2 * stddev)
        rand_init = rand_init * np.abs(1 - inf_mask) + inf_mask * (mean - 2 * stddev)
        sup_mask = rand_init > (mean + 2 * stddev)
        rand_init = rand_init * np.abs(1 - sup_mask) + sup_mask * (mean + 2 * stddev)
        return rand_init
