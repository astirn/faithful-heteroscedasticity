import os
import json
import zlib

import pandas as pd
import tensorflow as tf


class ZScoreNormalization(object):
    def __init__(self, y_mean, y_var):
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)

    def normalize_targets(self, y):
        return (y - self.y_mean) / self.y_std

    def scale_parameters(self, name, value):
        if name == 'loc':
            return value * self.y_std + self.y_mean
        elif name == 'scale':
            return value * self.y_std
        elif name == 'var':
            return value * self.y_var
        else:
            return value


def pretty_model_name(model, model_kwargs):
    if model.name == 'BetaNLL':
        name = 'Beta NLL'
        if 'beta' in model_kwargs.keys():
            name = name + ' (' + str(model_kwargs['beta']) + ')'
    elif model.name == 'Proposal1Normal':
        name = 'Proposal 1 Normal'
    elif model.name == 'Proposal2Normal':
        name = 'Proposal 2 Normal'
    elif 'VBEM' in model.name:
        name = 'VBEM*'
    else:
        name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()

    return name


def model_config_index(model_name, model_class, **kwargs):
    index_dict = {'Model': model_name.replace(' ' + model_class, ''), 'Class': model_class}
    for key, value in kwargs.items():
        value = value.__name__ if callable(value) else str(value)
        index_dict.update({key: value})
    index = pd.MultiIndex.from_tuples([tuple(index_dict.values())], names=list(index_dict.keys()))
    return index, str(str(index_dict))


def model_config_dir(base_path, model, model_kwargs, nn_kwargs):
    kwargs = {**model_kwargs, **nn_kwargs}
    for key, value in kwargs.items():
        value = value.__name__ if callable(value) else str(value)
        kwargs.update({key: value})
    config_dir = str(zlib.crc32(json.dumps(kwargs).encode('utf-8')))
    return os.path.join(base_path, model.name, config_dir)
