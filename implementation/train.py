from __future__ import print_function
import tensorflow as tf
import numpy as np

import sys
import os
import math
from plumbing import load_dataset, load_word_embeddings, dataset_preprocess

from networks import TensorflowTrainable, RNN, LSTMCell, AttentionLSTMCell
from batching import DataBatcher


project_root = os.path.dirname(os.getcwd())

parameters = {
    "runs_dir": os.path.join(project_root, 'runs'),
    "dataset_directory": os.path.join(project_root, 'snli_1.0'),
    "embeddings_path": os.path.join(project_root, 'GNews-Vectors.bin'),
    "model_name": "attention_lstm",
}

batch_parameters = {
    "batch_size_train": 24,
    "batch_size_dev": 10000,
    "batch_size_test": 10000
}

training_parameters = {
    "learning_rate": 0.001,
    "weight_decay": 0.0,
    "keep_prob": 0.8,
    "batch_size": {"train": batch_parameters['batch_size_train'], 
		   "dev": batch_parameters['batch_size_dev'],
	 	   "test": batch_parameters['batch_size_test']
		  },
    "gpu": 0,  # set to empty string to use CPU
    "num_epochs": 45,
    "embedding_dim": 300,
    "sequence_length": 20,
    "num_units": 100  # LSTM output dimension (k in the original paper)
}

parameters.update(training_parameters)

# #### Load Dataset + Pre-trained Embeddings


dataset = dataset_preprocess(load_dataset(parameters['dataset_directory']))
embeddings = load_word_embeddings(parameters['embeddings_path'])


def setup_logging(model_dir):
    logdir = os.path.join(model_dir, "log")
    if not os.path.exists(logdir):
        os.mkdir(logdir)
    logdir_train = os.path.join(logdir, "train")
    if not os.path.exists(logdir_train):
        os.mkdir(logdir_train)
    logdir_test = os.path.join(logdir, "test")
    if not os.path.exists(logdir_test):
        os.mkdir(logdir_test)
    logdir_dev = os.path.join(logdir, "dev")
    if not os.path.exists(logdir_dev):
        os.mkdir(logdir_dev)

    print("Logging setup done")

    return logdir_train, logdir_test, logdir_dev


def train(word_embeddings, dataset, parameters):
    modeldir = os.path.join(parameters["runs_dir"], parameters["model_name"])
    if not os.path.exists(modeldir):
        os.mkdir(modeldir)

    logdir_train, logdir_test, logdir_dev = setup_logging(modeldir)

    savepath = os.path.join(modeldir, "save")

    device_string = "/gpu:{}".format(parameters["gpu"]) if parameters["gpu"] else "/cpu:0"

    with tf.device(device_string):
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
        config_proto = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)

        sess = tf.Session(config=config_proto)

        premises_ph = tf.placeholder(tf.float32, shape=[parameters["sequence_length"], None, parameters["embedding_dim"]], name="premises")
        hypothesis_ph = tf.placeholder(tf.float32, shape=[parameters["sequence_length"], None, parameters["embedding_dim"]], name="hypothesis")
        targets_ph = tf.placeholder(tf.int32, shape=[None], name="targets")
        keep_prob_ph = tf.placeholder(tf.float32, name="keep_prob")

        _projecter = TensorflowTrainable()
        projecter = _projecter.get_4Dweights(filter_height=1,
                                             filter_width=parameters["embedding_dim"],
                                             in_channels=1, out_channels=parameters["num_units"],
                                             name="projecter")

        optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"],
                                           name="ADAM",
                                           beta1=0.9,
                                           beta2=0.999)

        with tf.variable_scope(name_or_scope="premise"):
            premise = RNN(cell=LSTMCell,
                          num_units=parameters["num_units"],
                          embedding_dim=parameters["embedding_dim"],
                          projecter=projecter,
                          keep_prob=keep_prob_ph)
            premise.process(sequence=premises_ph)

        with tf.variable_scope(name_or_scope="hypothesis"):
            hypothesis = RNN(cell=AttentionLSTMCell,
                             num_units=parameters["num_units"],
                             embedding_dim=parameters["embedding_dim"],
                             hiddens=premise.hiddens,
                             states=premise.states,
                             projecter=projecter,
                             keep_prob=keep_prob_ph)
            hypothesis.process(sequence=hypothesis_ph)

        loss, loss_summary, accuracy, accuracy_summary = hypothesis.loss(targets=targets_ph)

        weight_decay = tf.reduce_sum([tf.reduce_sum(parameter) for parameter in premise.parameters + hypothesis.parameters])

        global_loss = loss + (parameters["weight_decay"] * weight_decay)

        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        train_summary_writer = tf.summary.FileWriter(logdir_train, sess.graph)
        test_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        test_summary_writer = tf.summary.FileWriter(logdir_test)

        saver = tf.train.Saver(max_to_keep=10)
        # summary_writer = tf.train.SummaryWriter(logdir)
        tf.train.write_graph(sess.graph_def, modeldir, "graph.pb", as_text=False)
        # loader = tf.train.Saver(tf.all_variables())

        optimizer = tf.train.AdamOptimizer(learning_rate=parameters["learning_rate"], name="ADAM", beta1=0.9, beta2=0.999)
        train_op = optimizer.minimize(global_loss)

        sess.run(tf.global_variables_initializer())

        batcher = DataBatcher(word_embeddings)
        train_batches = batcher.batch_generator(dataset=dataset["train"], num_epochs=parameters["num_epochs"], batch_size=parameters["batch_size"]["train"], seq_length=parameters["sequence_length"])
        num_step_by_epoch = int(math.ceil(len(dataset["train"]["targets"]) / parameters["batch_size"]["train"]))

        for train_step, (train_batch, epoch) in enumerate(train_batches):
            feed_dict = {
                premises_ph: np.transpose(train_batch["premises"], (1, 0, 2)),
                hypothesis_ph: np.transpose(train_batch["hypothesis"], (1, 0, 2)),
                targets_ph: train_batch["targets"],
                keep_prob_ph: parameters["keep_prob"],
            }

            _, summary_str, train_loss, train_accuracy = sess.run([train_op, train_summary_op, loss, accuracy], feed_dict=feed_dict)
            train_summary_writer.add_summary(summary_str, train_step)

            if train_step % 100 is 0:
                sys.stdout.write("\rTRAIN | epoch={0}/{1}, step={2}/{3} | loss={4:.2f}, accuracy={5:.2f}%   ".format(epoch + 1, parameters["num_epochs"], train_step % num_step_by_epoch, num_step_by_epoch, train_loss, 100. * train_accuracy))
                sys.stdout.flush()

            if train_step % 5000 is 0:
                test_batches = batcher.batch_generator(dataset=dataset["test"], num_epochs=1, batch_size=parameters["batch_size"]["test"], seq_length=parameters["sequence_length"])
                for test_step, (test_batch, _) in enumerate(test_batches):
                    feed_dict = {
                        premises_ph: np.transpose(test_batch["premises"], (1, 0, 2)),
                        hypothesis_ph: np.transpose(test_batch["hypothesis"], (1, 0, 2)),
                        targets_ph: test_batch["targets"],
                        keep_prob_ph: 1.,
                    }

                    summary_str, test_loss, test_accuracy = sess.run([test_summary_op, loss, accuracy], feed_dict=feed_dict)
                    print("\nTEST | loss={0:.2f}, accuracy={1:.2f}%   ".format(test_loss, 100. * test_accuracy))
                    print()
                    test_summary_writer.add_summary(summary_str, train_step)
                    break

            if train_step % 5000 is 0:
                saver.save(sess, save_path=savepath, global_step=train_step)

        print()


if __name__ == '__main__':
    train(embeddings, dataset, parameters)
