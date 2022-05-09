import tensorflow as tf
from keras.utils import losses_utils


class Mean(tf.keras.metrics.Metric):
    def __init__(self, name, dtype=None):
        super(Mean, self).__init__(name, dtype=dtype)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, values, **kwargs):
        sum_values = tf.cast(tf.reduce_sum(values), self._dtype)
        num_values = tf.cast(tf.size(values), self._dtype)
        with tf.control_dependencies([sum_values, num_values]):
            self.total.assign_add(sum_values)
            self.count.assign_add(num_values)


class MeanLogLikelihood(Mean):
    def __init__(self, dtype=None):
        super(MeanLogLikelihood, self).__init__('LL', dtype=dtype)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        ll = tf.cast(y_pred[:, 2], self._dtype)
        return super(MeanLogLikelihood, self).update_state(ll, sample_weight=sample_weight)

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)


class RootMeanSquaredError(Mean):
    def __init__(self, dtype=None):
        super(RootMeanSquaredError, self).__init__('RMSE', dtype=dtype)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred[:, 0], self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        sq_err = tf.math.squared_difference(y_pred, y_true)
        return super(RootMeanSquaredError, self).update_state(values=sq_err, sample_weight=sample_weight)

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))
