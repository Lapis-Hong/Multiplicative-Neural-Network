#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/2/9
"""
This module contains TensorFlow Custom Estimator for MNN Model, based on tf.estimator.DNNClassifier.

There are two ways to build custom estimator.
    1. Write model_fn function to pass `tf.estimator.Estimator` to generate an instance.
        easier to build but with less flexibility. 
    2. Write subclass of `tf.estimator.Estimator` like premade(canned) estimators.
        much suitable for official project. 
Here, we use the second way to rewrite the original code of tf.estimator.DNNClassifier into tf.estimator.MNNClassifier.
"""
import six
import tensorflow as tf
from tensorflow.python.estimator.canned import head as head_lib

from mnn.util import add_layer_summary, _check_no_sync_replicas_optimizer, get_optimizer_instance

# The default learning rates are a historical artifact of the initial implementation.
_LEARNING_RATE = 0.05


def _mnn_logit_fn(features, mode, model_id, units, hidden_units, feature_columns,
                  activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Multiplicative Neural Network (MNN) logit_fn.
    Args:
        features: This is the first item returned from the `input_fn`
            passed to `train`, `evaluate`, and `predict`. This should be a
            single `Tensor` or `dict` of same.
        mode: Optional. Specifies if this training, evaluation or prediction. See
            `ModeKeys`.
        model_id: An int indicating the model index of Multi DNN.
        units: An int indicating the dimension of the logit layer.  In the
            MultiHead case, this should be the sum of all component Heads' logit
            dimensions.
        hidden_units: Iterable of integer number of hidden units per layer.
        feature_columns: Iterable of `feature_column._FeatureColumn` model inputs.
        activation_fn: Activation function applied to each layer.
        dropout: When not `None`, the probability we will drop out a given coordinate.
        batch_norm: Bool, Whether to use BN in dnn.
        input_layer_partitioner: Partitioner for input layer.
    Returns:
        A `Tensor` representing the logits, or a list of `Tensor`'s representing
      multiple logits in the MultiHead case.
    """
    with tf.variable_scope(
            'input_from_feature_columns',
            values=tuple(six.itervalues(features)),
            partitioner=input_layer_partitioner,
            reuse=tf.AUTO_REUSE):
        net = tf.feature_column.input_layer(
            features=features,
            feature_columns=feature_columns)

    with tf.variable_scope(
            'dnn_{}/hiddenlayer_{}'.format(model_id, "l"), values=(net,)) as hidden_layer_scope:
        lnet = tf.layers.dense(
            net,
            units=hidden_units[0],
            activation=activation_fn,
            kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
            name=hidden_layer_scope)

    with tf.variable_scope(
            'dnn_{}/hiddenlayer_{}'.format(model_id, "r"), values=(net,)) as hidden_layer_scope:
        rnet = tf.layers.dense(
            net,
            units=hidden_units[0],
            activation=activation_fn,
            kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
            name=hidden_layer_scope)
    hidden_units = hidden_units[1:]
    net = tf.multiply(lnet, rnet)  # 1st hidden layer
    # net = tf.concat([tf.multiply(lnet, rnet), lnet, rnet], axis=1)  # 1st hidden layer

    for layer_id, num_hidden_units in enumerate(hidden_units):
        with tf.variable_scope(
                'dnn_{}/hiddenlayer_{}'.format(model_id, layer_id+1), values=(net,)) as hidden_layer_scope:
            net = tf.layers.dense(
                net,
                units=num_hidden_units,
                activation=activation_fn,
                kernel_initializer=tf.glorot_uniform_initializer(),  # also called Xavier uniform initializer.
                name=hidden_layer_scope)

            # Add dropout and BN.
            if dropout is not None and mode == tf.estimator.ModeKeys.TRAIN:
                net = tf.layers.dropout(net, rate=dropout, training=True)  # dropout rate
            if batch_norm:
                net = tf.layers.batch_normalization(net)  # add bn layer, it has been added in high version tf
        add_layer_summary(net, hidden_layer_scope.name)

    with tf.variable_scope('dnn_{}/logits'.format(model_id), values=(net,)) as logits_scope:
        logits = tf.layers.dense(
                net,
                units=units,
                kernel_initializer=tf.glorot_uniform_initializer(),
                name=logits_scope)
    add_layer_summary(logits, logits_scope.name)

    return logits


def multimnn_logit_fn_builder(units, hidden_units_list, feature_columns,
                              activation_fn, dropout, batch_norm, input_layer_partitioner):
    """Multi MNN logit function builder.
    Args:
        hidden_units_list: 1D iterable obj for single DNN or 2D for Multi DNN.
            eg: [128, 64, 32] or [[128, 64, 32], [64, 32]]
    Returns:
        Multi DNN logit fn.
    """
    if not isinstance(units, int):
        raise ValueError('units must be an int. Given type: {}'.format(type(units)))

    if not isinstance(hidden_units_list[0], (list, tuple)):
        hidden_units_list = [hidden_units_list]

    def multimnn_logit_fn(features, mode):
        logits = []
        for idx, hidden_units in enumerate(hidden_units_list):
            logits.append(
                _mnn_logit_fn(
                    features,
                    mode,
                    idx + 1,
                    units,
                    hidden_units,
                    feature_columns,
                    activation_fn,
                    dropout,
                    batch_norm,
                    input_layer_partitioner))
        logits = tf.add_n(logits)  # Adds all input tensors element-wise.

        return logits

    return multimnn_logit_fn


def _mnn_model_fn(
        features, labels, mode, head,
        feature_columns=None,
        optimizer='Adagrad',
        hidden_units=None,
        activation_fn=tf.nn.relu,
        dropout=None,
        batch_norm=None,
        input_layer_partitioner=None,
        config=None):
    """MNN model_fn.
    Args:
        features: dict of `Tensor`.
        labels: `Tensor` of shape [batch_size, 1] or [batch_size] labels of dtype
            `int32` or `int64` in the range `[0, n_classes)`.
      mode: Defines whether this is training, evaluation or prediction. See `ModeKeys`.
      head: A `Head` instance.
      model_type: one of `wide`, `deep`, `wide_deep`.
      feature_columns: An iterable containing all the feature columns used by
        the DNN model.
      optimizer: String, `Optimizer` object, or callable that defines the
        optimizer to use for training the DNN model. Defaults to the Adagrad
        optimizer.
      hidden_units: List of hidden units per DNN layer, nested lists for Multi DNN.
      activation_fn: Activation function applied to each DNN layer. If `None`,
          will use `tf.nn.relu`.
      dropout: When not `None`, the probability we will drop out a given DNN
          coordinate.
      batch_norm: Bool, add BN layer after each DNN layer
      input_layer_partitioner: Partitioner for input layer.
          config: `RunConfig` object to configure the runtime settings.
    Returns:
        `ModelFnOps`
    Raises:
        ValueError: If `dnn_features_columns` are empty 
            or `input_layer_partitioner` is missing, or features has the wrong type.
    """
    if not isinstance(features, dict):
        raise ValueError('features should be a dictionary of `Tensor`s. '
                         'Given type: {}'.format(type(features)))
    num_ps_replicas = config.num_ps_replicas if config else 0
    input_layer_partitioner = input_layer_partitioner or (
        tf.min_max_variable_partitioner(max_partitions=num_ps_replicas,
                                        min_slice_size=64 << 20))
    # Build MNN Logits.
    mnn_parent_scope = 'mnn'

    optimizer = get_optimizer_instance(
        optimizer, learning_rate=_LEARNING_RATE)

    mnn_partitioner = tf.min_max_variable_partitioner(max_partitions=num_ps_replicas)
    with tf.variable_scope(
            mnn_parent_scope,
            values=tuple(six.itervalues(features)),
            partitioner=mnn_partitioner):
        mnn_logit_fn = multimnn_logit_fn_builder(
            units=head.logits_dimension,
            hidden_units_list=hidden_units,
            feature_columns=feature_columns,
            activation_fn=activation_fn,
            dropout=dropout,
            batch_norm=batch_norm,
            input_layer_partitioner=input_layer_partitioner
        )
        logits = mnn_logit_fn(features=features, mode=mode)

    def _train_op_fn(loss):
        """Returns the op to optimize the loss."""
        global_step = tf.train.get_global_step()
        # BN, when training, the moving_mean and moving_variance need to be updated. By default the
        # update ops are placed in tf.GraphKeys.UPDATE_OPS, so they need to be added as a dependency to the train_op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=mnn_parent_scope))

        with tf.control_dependencies([train_op]):
            # Returns a context manager that specifies an op to colocate with.
            with tf.colocate_with(global_step):
                return tf.assign_add(global_step, 1)

    return head.create_estimator_spec(
        features=features,
        mode=mode,
        labels=labels,
        train_op_fn=_train_op_fn,
        logits=logits)


class MNNClassifier(tf.estimator.Estimator):
    """TensorFlow tf.estimator.MNNClassifier.
    The usage and behavior is exactly same with tf.estimator.DNNClassifier.
    """

    def __init__(self,
                 model_dir=None,
                 feature_columns=None,
                 optimizer='Adagrad',
                 hidden_units=None,
                 activation_fn=tf.nn.relu,
                 dropout=None,
                 batch_norm=None,
                 n_classes=2,
                 weight_column=None,
                 label_vocabulary=None,
                 input_layer_partitioner=None,
                 config=None):
        """Initializes a WideDeepCombinedClassifier instance.
        Args:
            model_dir: Directory to save model parameters, graph and etc. This can
                also be used to load checkpoints from the directory into a estimator
                to continue training a previously saved model.
            feature_columns: An iterable containing all the feature columns used
                by deep part of the model. All items in the set must be instances of
                classes derived from `FeatureColumn`.
            optimizer: An instance of `tf.Optimizer` used to apply gradients to
                the deep part of the model. Defaults to Adagrad optimizer.
            hidden_units: List of hidden units per layer. All layers are fully
                connected.
            activation_fn: Activation function applied to each layer. If None,
                will use `tf.nn.relu`.
            dropout: When not None, the probability we will drop out
                a given coordinate.
            n_classes: Number of label classes. Defaults to 2, namely binary
                classification. Must be > 1.
            weight_column: A string or a `_NumericColumn` created by
                `tf.feature_column.numeric_column` defining feature column representing
                weights. It is used to down weight or boost examples during training. It
                will be multiplied by the loss of the example. If it is a string, it is
                used as a key to fetch weight tensor from the `features`. If it is a
                `_NumericColumn`, raw tensor is fetched by key `weight_column.key`,
                then weight_column.normalizer_fn is applied on it to get weight tensor.
            label_vocabulary: A list of strings represents possible label values. If
                given, labels must be string type and have any value in
                `label_vocabulary`. If it is not given, that means labels are
                already encoded as integer or float within [0, 1] for `n_classes=2` and
                encoded as integer values in {0, 1,..., n_classes-1} for `n_classes`>2 .
                Also there will be errors if vocabulary is not provided and labels are
                string.
            input_layer_partitioner: Partitioner for input layer. Defaults to
                `min_max_variable_partitioner` with `min_slice_size` 64 << 20.
            config: RunConfig object to configure the runtime settings.
        Raises:
            ValueError: If both linear_feature_columns and dnn_features_columns are
                empty at the same time.
        """
        if not feature_columns:
            raise ValueError('Feature columns must be defined.')
        if not hidden_units:
            raise ValueError('Hidden units must be defined.')

        if n_classes == 2:
            # units = 1
            head = head_lib._binary_logistic_head_with_sigmoid_cross_entropy_loss(
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)
        else:
            # units = n_classes
            head = head_lib._multi_class_head_with_softmax_cross_entropy_loss(
                n_classes,
                weight_column=weight_column,
                label_vocabulary=label_vocabulary)

        def _model_fn(features, labels, mode, config):
            return _mnn_model_fn(
                features=features,
                labels=labels,
                mode=mode,
                head=head,
                feature_columns=feature_columns,
                optimizer=optimizer,
                hidden_units=hidden_units,
                activation_fn=activation_fn,
                dropout=dropout,
                batch_norm=batch_norm,
                input_layer_partitioner=input_layer_partitioner,
                config=config)
        super(MNNClassifier, self).__init__(
            model_fn=_model_fn, model_dir=model_dir, config=config)
