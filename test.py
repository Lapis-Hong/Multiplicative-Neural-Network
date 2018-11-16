#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""MNN Model Evaluation"""
import argparse
import os
import sys
import time

import tensorflow as tf

from mnn.build_estimator import build_estimator
from mnn.dataset import input_fn
from mnn.read_conf import Config
from mnn.util import elapse_time

CONFIG = Config("conf/criteo")
# CONFIG = Config("conf/avazu")

parser = argparse.ArgumentParser(description='Evaluate MNN Model.')

parser.add_argument(
    '--conf_dir', type=bool, default="conf/criteo",
    help='Path to configuration.')

parser.add_argument(
    '--test_data', type=str, default=CONFIG.train["test_data"],
    help='Evaluating data dir.')

parser.add_argument(
    '--model_dir', type=str, default=CONFIG.train["model_dir"],
    help='Model checkpoint dir for evaluating.')

parser.add_argument(
    '--model_type', type=str, default=CONFIG.train["model_type"],
    help="Valid model types: {'wide', 'deep', 'wide_deep'}.")

parser.add_argument(
    '--batch_size', type=int, default=CONFIG.train["batch_size"],
    help='Number of examples per batch.')

parser.add_argument(
    '--checkpoint_path', type=str, default=CONFIG.train["checkpoint_path"],
    help="Path of a specific checkpoint to evaluate. If None, the latest checkpoint in model_dir is used.")


def main(_):
    print("Using TensorFlow version %s, need TensorFlow 1.10 or later." % tf.__version__)
    print('Model directory: {}'.format(FLAGS.model_dir))
    model = build_estimator(FLAGS.model_dir, FLAGS.model_type)
    tf.logging.info('Build estimator: {}'.format(model))

    tf.logging.info('='*30+' START TESTING'+'='*30)
    s_time = time.time()
    results = model.evaluate(input_fn=lambda: input_fn(FLAGS.test_data, 1, FLAGS.batch_size, False),
                             steps=None,  # Number of steps for which to evaluate model.
                             hooks=None,
                             checkpoint_path=FLAGS.checkpoint_path,  # If None, the latest checkpoint is used.
                             name=None)
    tf.logging.info('='*30+'FINISH TESTING, TAKE {}'.format(elapse_time(s_time))+'='*30)
    # Display evaluation metrics
    print('-' * 80)
    for key in sorted(results):
        print('%s: %s' % (key, results[key]))

if __name__ == '__main__':
    # Set to INFO for tracking training, default is WARN. ERROR for least messages
    tf.logging.set_verbosity(tf.logging.INFO)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
