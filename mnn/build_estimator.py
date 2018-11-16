#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/15
"""This module for building estimator for tf.estimators API."""
import numpy as np
import tensorflow as tf

from mnn.model import MNNClassifier

# feature columns
categorical_column_with_hash_bucket = tf.feature_column.categorical_column_with_hash_bucket
numeric_column = tf.feature_column.numeric_column
embedding_column = tf.feature_column.embedding_column


def _embed_size(size):
    """Empirical embedding dim."""
    # return 6 * dim ** 0.25  # 6×(category cardinality)1/4
    return int(np.power(2, np.ceil(np.log(size ** 0.4))))


def _hash_bucket_size(size):
    if size <= 100:
        return 4 * size
    elif 100 < size <= 10000:
        return 3 * size
    else:
        return 2 * size


def _build_model_columns(feature_conf):
    """
    Build mnn feature columns.
        columns: category features + discretized continuous features
    Args:
        feature_conf: Feature configuration dict (Config instance feature attribute).
    Return: 
        _DenseColumn in tf.estimators API
    """
    tf.logging.info('Total feature classes: {}'.format(len(feature_conf)))
    columns = []
    input_dim = 0

    for feature, conf in feature_conf.items():
        if conf["type"] == 'category':  # category features
            f_size = conf["size"]
            hash_bucket_size, embedding_size = _hash_bucket_size(f_size), _embed_size(f_size)
            col = categorical_column_with_hash_bucket(feature, hash_bucket_size=hash_bucket_size)
            columns.append(embedding_column(col, dimension=embedding_size, combiner='mean'))
            input_dim += embedding_size

        else:  # continuous features
            mean, std = conf["mean"], conf["std"]
            col = numeric_column(feature, shape=(1,), normalizer_fn=lambda x: (x-mean) / std)
            columns.append(col)
            input_dim += 1
    # Add columns logging info
    tf.logging.info('Build total {} deep columns'.format(len(columns)))
    for col in columns:
        tf.logging.debug('Columns: {}'.format(col))
    tf.logging.info('MNN input dimension is: {}'.format(input_dim))

    return columns


def _build_opt(opt, lr, l1, l2, lr_decay, lr_decay_steps=10000, lr_decay_rate=0.96):
    if lr_decay:
        return lambda: opt(
            learning_rate=tf.train.exponential_decay(
                learning_rate=lr,
                global_step=tf.train.get_global_step(),
                decay_steps=lr_decay_steps,
                decay_rate=lr_decay_rate),
            l1_regularization_strength=l1,
            l2_regularization_strength=l2)
    else:
        return opt(
            learning_rate=lr,
            l1_regularization_strength=l1,
            l2_regularization_strength=l2)


def build_estimator(model_dir, conf):
    """Build an estimator for mnn model."""
    columns = _build_model_columns(conf.feature)

    # Create a tf.estimator.RunConfig to ensure the model is run on CPU, which
    # trains faster than GPU for this model.
    run_config = tf.estimator.RunConfig(**conf.runconfig).replace(
        session_config=tf.ConfigProto(device_count={'GPU': 0}))

    conf = conf.model
    # Optimizer with regularization and learning rate decay.
    deep_opt = _build_opt(
        tf.train.ProximalAdagradOptimizer,
        conf["learning_rate"], conf["l1"], conf["l2"],
        conf["lr_decay"], conf["lr_decay_steps"], conf["lr_decay_rate"])

    # deep_opt = tf.train.AdamOptimizer(
    #     learning_rate=0.001,
    #     beta1=0.9,
    #     beta2=0.999,
    #     epsilon=1e-08,
    #     use_locking=False,
    #     name='Adam')

    return MNNClassifier(
        model_dir=model_dir,
        feature_columns=columns,
        optimizer=deep_opt,
        hidden_units=conf["hidden_units"],
        activation_fn=eval(conf["activation_function"]),
        dropout=conf["dropout"],
        batch_norm=conf["batch_normalization"],
        n_classes=2,
        weight_column=None,
        label_vocabulary=None,
        input_layer_partitioner=None,
        config=run_config)

if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.DEBUG)
    from mnn.read_conf import Config
    conf = Config("../conf/criteo")
    model = build_estimator('./mnt_model', conf)
    # print(model.config)  # <tensorflow.python.estimator.run_config.RunConfig object at 0x118de4e10>
    # print(model.model_dir)  # ./model
    # print(model.model_fn)  # <function public_model_fn at 0x118de7b18>
    # print(model.params)  # {}
    # print(model.get_variable_names())
    # print(model.get_variable_value('dnn/hiddenlayer_0/bias'))
    # print(model.latest_checkpoint())  # another 4 method is export_savedmodel,train evaluate predict
