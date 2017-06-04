# coding: utf-8
import tensorflow as tf
import numpy as np

from networks.base import TensorflowTrainable


class RNN(TensorflowTrainable):
    def __init__(self, cell, num_units, embedding_dim, projecter, keep_prob, **kwargs):
        super(RNN, self).__init__()

        # private
        self._projecter = projecter
        self._embedding_dim = embedding_dim
        self._num_units = num_units
        self._cell = cell(num_units=self._num_units, **kwargs)
        self.keep_prob = keep_prob

        # public
        self.predictions = None
        self.hiddens = None
        self.states = None

    def process(self, sequence):
        noisy_sequence = tf.nn.dropout(x=sequence, keep_prob=self.keep_prob, name="noisy_inputs")
        noisy_sequence = tf.expand_dims(tf.transpose(noisy_sequence, [1, 0, 2]), 3)
        projected_sequence = tf.transpose(tf.squeeze(tf.nn.conv2d(input=noisy_sequence, filter=self._projecter, strides=[1, 1, 1, 1], padding="VALID"), [2]), [1, 0, 2])

        list_sequence = tf.unstack(projected_sequence)
        self._cell.initialize_something(input=list_sequence[0])
        for i, input in enumerate(list_sequence):
            self._cell.process(input=input, delimiter=i==0)
        self.states, self.hiddens = self._cell.c[1:], self._cell.h[1:]

    def get_predictions(self):
        biases = self.get_biases(dim_out=3, name="biases")
        weights = self.get_weights(dim_in=self._num_units, dim_out=3, name="weights")
        noisy_features = tf.nn.dropout(x=self._cell.features, keep_prob=self.keep_prob, name="noisy_features")
        self.predictions = tf.transpose(tf.add(tf.matmul(weights, noisy_features), biases), [1, 0])
        return self.predictions

    def loss(self, targets):
        if self.hiddens is None:
            raise Exception("You shouldn't have been there.")
        else:
            with tf.name_scope("loss") as scope:
                loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.get_predictions(), labels=targets))
                loss_summary = tf.summary.scalar("loss", loss)
            with tf.name_scope("accuracy") as scope:
                predictions = tf.to_int32(tf.argmax(self.predictions, 1))
                correct_label = tf.to_int32(targets)
                correct_predictions = tf.equal(predictions, correct_label)
                accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"))
                accuracy_summary = tf.summary.scalar("accuracy", accuracy)
            return loss, loss_summary, accuracy, accuracy_summary
