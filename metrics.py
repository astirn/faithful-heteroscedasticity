import tensorflow as tf
import tensorflow_probability as tfp


def pack_predictor_values(mean, ll, cdf):
    return tf.concat([mean, ll, cdf], axis=-1)


def unpack_predictor_values(predictor_values, request):
    predictor_values = tf.split(predictor_values, num_or_size_splits=3, axis=-1)
    if request == 'mean':
        return predictor_values[0]
    elif request == 'll':
        return tf.keras.layers.Flatten()(predictor_values[1])
    elif request == 'cdf':
        return tf.keras.layers.Flatten()(predictor_values[2])
    else:
        raise NotImplementedError


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
        ll = tf.cast(tf.reduce_sum(unpack_predictor_values(y_pred, 'll'), axis=-1), self._dtype)
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
        mean = tf.cast(unpack_predictor_values(y_pred, 'mean'), self._dtype)
        sq_err = tf.reduce_sum(tf.keras.layers.Flatten()(tf.math.squared_difference(y_true, mean)), axis=1)
        return super(RootMeanSquaredError, self).update_state(values=sq_err, sample_weight=sample_weight)

    def result(self):
        return tf.sqrt(tf.math.divide_no_nan(self.total, self.count))


class ExpectedCalibrationError(Mean):
    def __init__(self, num_bins=6, dtype=None):
        super(ExpectedCalibrationError, self).__init__('ECE', dtype=dtype)
        self.edges = tf.stack([x / num_bins for x in range(num_bins + 1)])
        self.probs = self.edges[1:] - self.edges[:-1]
        self.bin_counts = [self.add_weight('count_{:d}'.format(i), initializer='zeros') for i in range(num_bins)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        cdf = tf.cast(unpack_predictor_values(y_pred, 'cdf'), self._dtype)
        cdf = tf.reshape(cdf, [-1])
        sum_values = tf.split(tfp.stats.histogram(cdf, self.edges), len(self.bin_counts))
        sum_values = [tf.squeeze(x) for x in sum_values]
        with tf.control_dependencies(sum_values):
            for i, bin_count in enumerate(self.bin_counts):
                bin_count.assign_add(sum_values[i])

    def result(self):
        bin_counts = tf.stack(self.bin_counts)
        bin_counts /= tf.reduce_sum(bin_counts)
        return tf.reduce_sum((self.probs - bin_counts) ** 2)
