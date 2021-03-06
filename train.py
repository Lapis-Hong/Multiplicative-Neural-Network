#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""Training MNN Model."""
import argparse
import shutil
import sys
import time

import tensorflow as tf

from mnn.build_estimator import build_estimator
from mnn.dataset import input_fn
from mnn.read_conf import Config
from mnn.util import elapse_time

# Config file path, change it to use different data.
CONFIG = Config("conf/criteo")
# CONFIG = Config("conf/avazu")

parser = argparse.ArgumentParser(description='Train MNN Model.')

parser.add_argument(
    '--train_data', type=str, default=CONFIG.train["train_data"],
    help='Path to the train data.')
parser.add_argument(
    '--dev_data', type=str, default=CONFIG.train["dev_data"],
    help='Path to the validation data.')
parser.add_argument(
    '--model_dir', type=str, default=CONFIG.train["model_dir"],
    help='Base directory for the model.')
parser.add_argument(
    '--train_epochs', type=int, default=CONFIG.train["train_epochs"],
    help='Number of training epochs.')
parser.add_argument(
    '--epochs_per_eval', type=int, default=CONFIG.train["epochs_per_eval"],
    help='The number of training epochs to run between evaluations.')
parser.add_argument(
    '--steps_per_eval', type=int, default=CONFIG.train["steps_per_eval"],
    help='The number of training steps to run between evaluations.')
parser.add_argument(
    '--batch_size', type=int, default=CONFIG.train["batch_size"],
    help='Number of examples per batch.')
parser.add_argument(
    '--keep_train', type=int, default=CONFIG.train["keep_train"],
    help='Whether to keep training on previous trained model.')
parser.add_argument(
    '--num_samples', type=int, default=CONFIG.train["num_samples"],
    help='Number of samples used for shuffle buffer size.')
parser.add_argument(
    '--verbose', type=bool, default=CONFIG.train["verbose"],
    help='Set 0 for tf log level INFO, 1 for ERROR')


def train(model):
    """Custom train and eval function, eval between epochs."""
    tf.logging.info('Evaluate every {} epochs'.format(FLAGS.epochs_per_eval))
    best_auc, best_logloss, best_epoch = 0, 10000, 0  # saving best auc and logloss
    for n in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tf.logging.info('START TRAIN AT EPOCH {}'.format(FLAGS.epochs_per_eval*n + 1))
        t0 = time.time()
        model.train(input_fn=lambda: input_fn(CONFIG, FLAGS.train_data, FLAGS.epochs_per_eval, FLAGS.batch_size, True, FLAGS.num_samples),
                    hooks=None,
                    steps=None,
                    max_steps=None,
                    saving_listeners=None)
        tf.logging.info('Finish train {} epochs, take {} mins'.format(n + 1, elapse_time(t0)))
        print('-' * 80)

        t0 = time.time()
        results = model.evaluate(input_fn=lambda: input_fn(CONFIG, FLAGS.dev_data, 1, FLAGS.batch_size, False),
                                 steps=None,  # Number of steps for which to evaluate model.
                                 hooks=None,
                                 checkpoint_path=None,  # If None, the latest checkpoint in model_dir is used.
                                 name=None)
        tf.logging.info('Finish evaluation, take {} mins'.format(elapse_time(t0)))
        print('-' * 80)

        # Display evaluation metrics
        print('Evaluation metrics at epoch {}: (* means improve)'.format(n+1))
        improve_auc_token, improve_loss_token = "", ""
        for key in sorted(results):
            value = results[key]
            print('\t{}: {}'.format(key, value))
            if key == "auc" and value > best_auc:
                best_auc = value
                improve_auc_token = "*"
            elif key == "average_loss" and value < best_logloss:
                best_logloss = value
                improve_loss_token = "*"

        if improve_loss_token or improve_auc_token:
                best_epoch = n + 1
        print("\nMAX AUC={:.6f} {}\nMIN LOSS={:.6f} {}".format(
            best_auc, improve_auc_token, best_logloss, improve_loss_token))
        print('-' * 80)

        # Early stopping after 3 epoch no improvement.
        if n + 1 - best_epoch >= 3:
            exit("No improvement for 3 epochs, early stopping.")


def train_and_eval(model):
    """tf.estimator train and eval function, eval between steps."""
    train_spec = tf.estimator.TrainSpec(
        input_fn=lambda: input_fn(CONFIG, FLAGS.train_data, FLAGS.train_epochs, FLAGS.batch_size, True, FLAGS.num_samples),
        max_steps=None)
    eval_spec = tf.estimator.EvalSpec(
        input_fn=lambda: input_fn(CONFIG, FLAGS.dev_data, 1, FLAGS.batch_size, False),
        steps=FLAGS.steps_per_eval,
        start_delay_secs=1800
    )
    tf.estimator.train_and_evaluate(model, train_spec, eval_spec)


def main(_):
    print("Using TensorFlow version %s, need TensorFlow 1.10 or later." % tf.__version__)
    CONFIG.print_config()  # print config
    print('Model directory: {}'.format(FLAGS.model_dir))
    if not FLAGS.keep_train:
        # Clean up the model directory if not keep training
        shutil.rmtree(FLAGS.model_dir, ignore_errors=True)
        tf.logging.info('Remove model directory: {}'.format(FLAGS.model_dir))
    model = build_estimator(FLAGS.model_dir, CONFIG)
    tf.logging.info('Build estimator: {}'.format(model))

    if FLAGS.steps_per_eval:
        train_and_eval(model)
    else:
        train(model)  # local run
    train(model)

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    if FLAGS.verbose:
        # Set to INFO for tracking training, default is WARN. ERROR for least messages
        tf.logging.set_verbosity(tf.logging.ERROR)
    else:
        tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
