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
        if name == 'mean':
            return value * self.y_std + self.y_mean
        elif name == 'std':
            return value * self.y_std
        elif name == 'var':
            return value * self.y_var
        elif name == 'y':
            return value
        else:
            raise NotImplementedError

    # def de_whiten_precision(self, precision):
    #     return precision / self.y_var
    #
    # def de_whiten_log_precision(self, log_precision):
    #     return log_precision - tf.math.log(self.y_var)


def pretty_model_name(model, model_kwargs):
    name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
    name = name.replace('N L L', 'NLL')
    if len(model_kwargs) > 0:
        name = name + ' (' + ', '.join([str(v) for v in model_kwargs.values()]) + ')'
    return name


def model_config_index(model_name, kwargs):
    index = {'Model': model_name}
    for key, value in kwargs.items():
        value = value.__name__ if callable(value) else str(value)
        index.update({key: value})
    print('********** ' + str(index) + ' **********')
    index = pd.MultiIndex.from_tuples([tuple(index.values())], names=list(index.keys()))
    return index
