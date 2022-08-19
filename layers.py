import tensorflow as tf


class SHAPyCat(tf.keras.layers.Layer):
    def __init__(self, model):
        tf.keras.layers.Layer.__init__(self)
        self.f_trunk = model.f_trunk
        self.f_mean = model.f_mean
        self.f_scale = model.f_scale if hasattr(model, 'f_scale') else None

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        mean = self.f_mean(z, **kwargs)
        stddev = tf.ones_like(mean) if self.f_scale is None else self.f_scale(z, **kwargs)
        return tf.concat([mean, stddev], axis=-1)
