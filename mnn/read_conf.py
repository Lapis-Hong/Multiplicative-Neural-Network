#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: lapis-hong
# @Date  : 2018/1/24
"""Read All Configuration from ../conf/*.yaml"""
import os
import yaml

from mnn.util import check_file_exist


class Config(object):
    """Config class"""
    def __init__(self, config_dir):
        # self._base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self._schema_conf_file = os.path.join(config_dir, "schema.yaml")
        self._feature_conf_file = os.path.join(config_dir, "feature.yaml")
        self._train_conf_file = os.path.join(config_dir, "train.yaml")
        check_file_exist(self._schema_conf_file)
        check_file_exist(self._feature_conf_file)
        check_file_exist(self._train_conf_file)

    @staticmethod
    def _check_feature_conf(feature, schema, f_type):
        if feature not in schema:
            raise ValueError("Invalid feature name `{}` in feature.yaml, "
                             "must be consistent with schema.yaml".format(feature))
        if f_type is None:
            raise ValueError("feature type are required in feature conf, "
                             "found empty value for feature `{}`.".format(feature))
        assert f_type in {'category', 'continuous'}, (
            "Invalid type `{}` for feature `{}` in feature conf, "
            "must be `category` or `continuous`.".format(f_type, feature))

    def _read_schema_conf(self):
        with open(self._schema_conf_file) as f:
            return yaml.load(f)  # {k: v.lower() for k, v in yaml.load(f).items()}

    def _read_feature_conf(self):
        with open(self._feature_conf_file) as f:
            feature_conf = yaml.load(f)
            schema = self._read_schema_conf().values()
            for feature, conf in feature_conf.items():
                type_ = conf["type"]
                self._check_feature_conf(feature, schema, type_)
            return feature_conf

    def _read_train_conf(self):
        with open(self._train_conf_file) as f:
            return yaml.load(f)

    @property
    def schema(self):
        return self._read_schema_conf()

    @property
    def feature(self):
        return self._read_feature_conf()

    @property
    def config(self):
        return self._read_train_conf()

    @property
    def train(self):
        conf = self._read_train_conf()["train"]
        if conf["field_delim"] == r"\t":  # change \\t to \t
            conf["field_delim"] = "\t"
        return conf

    @property
    def model(self):
        return self._read_train_conf()["model"]

    @property
    def runconfig(self):
        return self._read_train_conf()["runconfig"]

    def print_config(self):
        conf = self._read_train_conf()
        print("\nParameters:")
        print("train:")
        for k, v in conf["train"].items():
            print("\t{}: {}".format(k, v))
        print("model:")
        for k, v in conf["model"].items():
            print("\t{}: {}".format(k, v))


if __name__ == '__main__':
    config_dir = '../conf/criteo'
    conf = Config(config_dir)
    print(conf.schema)
    print(conf.feature)
    print(conf.config)
    print(conf.train)
    print(conf.runconfig)
    print(conf.model)
    conf.print_config()






