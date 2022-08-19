import tensorflow as tf


class SHAPyCat(tf.keras.layers.Layer):
    def __init__(self, model, f_trunk, f_param, **kwargs):
        tf.keras.layers.Layer.__init__(self)

        self.f_trunk = f_trunk(d_in=model.f_trunk.input_shape[1:])
        dim_latent = self.f_trunk.output_shape[1:]
        assert len(dim_latent) == 1
        self.f_mean = f_param(d_in=dim_latent[0], d_out=1, f_out=None, name='f_mean', **kwargs)
        if hasattr(model, 'f_scale'):
            self.f_scale = f_param(d_in=dim_latent, d_out=1, f_out='softplus', name='f_scale', **kwargs)
        else:
            self.f_scale = lambda x, **k: None

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        mean = self.f_mean(z, **kwargs)
        stddev = self.f_scale(z, **kwargs) or tf.ones_like(mean)
        return tf.concat([mean, stddev], axis=-1)
