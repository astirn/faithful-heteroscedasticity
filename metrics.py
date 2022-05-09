import tensorflow as tf
from keras.utils import losses_utils


class LogLikelihood(tf.keras.metrics.Metric):
    def __init__(self, name='LL', dtype=None):
        super(LogLikelihood, self).__init__(name, dtype=dtype)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.total.assign_add(tf.reduce_sum(y_pred[:, 2]))

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))


class RootMeanSquaredError(tf.keras.metrics.Metric):
    def __init__(self, name='RMSE', dtype=None):
        super(RootMeanSquaredError, self).__init__(name, dtype=dtype)
        self.total = self.add_weight('total', initializer='zeros')
        self.count = self.add_weight('count', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, self._dtype)
        y_pred = tf.cast(y_pred[:, 0], self._dtype)
        y_pred, y_true = losses_utils.squeeze_or_expand_dimensions(y_pred, y_true)
        error_sq = tf.math.squared_difference(y_pred, y_true)
        value_sum = tf.reduce_sum(error_sq)
        with tf.control_dependencies([value_sum]):
            update_total_op = self.total.assign_add(value_sum)
        num_values = tf.cast(tf.size(error_sq), self._dtype)
        with tf.control_dependencies([update_total_op]):
            return self.count.assign_add(num_values)

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))
