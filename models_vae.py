import argparse

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from tensorflow_probability import distributions as tfpd

from callbacks import RegressionCallback
from metrics import pack_predictor_values, MeanLogLikelihood, RootMeanSquaredError, ExpectedCalibrationError
from data_vae import load_data


# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def encoder_dense(dim_in, hidden_dims, dim_out, name, **kwargs):
    enc = tf.keras.Sequential(name=name)
    enc.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    enc.add(tf.keras.layers.Flatten())
    for dim in hidden_dims:
        enc.add(tf.keras.layers.Dense(units=dim, activation='elu'))
    enc.add(tf.keras.layers.Dense(units=dim_out))
    return enc


def decoder_dense(dim_in, hidden_dims, dim_out, f_out, name, **kwargs):
    dec = tf.keras.Sequential(name=name)
    dec.add(tf.keras.Input(shape=dim_in, dtype=tf.float32))
    for dim in hidden_dims:
        dec.add(tf.keras.layers.Dense(units=dim, activation='elu'))
    dec.add(tf.keras.layers.Dense(units=dim_out, activation=f_out))
    return dec


class HeteroscedasticVariationalAutoencoder(tf.keras.Model):

    def __init__(self, dim_x, dim_z, encoder, num_mc_samples, **kwargs):
        assert isinstance(dim_x, (list, tuple))
        assert isinstance(dim_z, int) and dim_z > 0
        assert isinstance(num_mc_samples, int) and num_mc_samples > 0
        tf.keras.Model.__init__(self, name=kwargs.pop('name'))

        # save configuration
        self.dim_x = dim_x
        self.dim_z = dim_z
        self.num_mc_samples = num_mc_samples

        # prior for latent codes
        self.p_z = tfpd.Normal(loc=tf.zeros(dim_z), scale=tf.ones(dim_z))
        self.p_z = tfpd.Independent(self.p_z, reinterpreted_batch_ndims=1)

        # flatten layer
        self.flatten = tf.keras.layers.Flatten()

        # encoder network
        self.q_z = encoder(dim_x, [512, 256, 128], 2 * dim_z, name='q_z', **kwargs)
        self.q_z.add(tfp.layers.IndependentNormal(dim_z))

    @staticmethod
    def parse_keras_inputs(data):
        if isinstance(data, dict):
            x = data['x']
        elif isinstance(data, tuple):
            x = data
        else:
            raise NotImplementedError
        return x

    def train_step(self, data):
        x = self.parse_keras_inputs(data)

        # optimization step
        with tf.GradientTape() as tape:
            loss, *params = self.call(x)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        # update metrics
        self.update_metrics(x, *params)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = self.parse_keras_inputs(data)

        # update metrics
        self.update_metrics(x, *self.call(x, training=True)[1:])

        return {m.name: m.result() for m in self.metrics}


class NormalVAE(HeteroscedasticVariationalAutoencoder):

    def __init__(self, dim_x, dim_z, decoder, **kwargs):
        HeteroscedasticVariationalAutoencoder.__init__(self, dim_x, dim_z, name='NormalVAE', **kwargs)

        # decoder networks
        self.mean = decoder(dim_z, [128, 256, 512], np.prod(dim_x), f_out=None, name='mu', **kwargs)
        self.precision = decoder(dim_z, [128, 256, 512], np.prod(dim_x), f_out='softplus', name='lambda', **kwargs)

    def call(self, x, **kwargs):

        # variational family and its KL-divergence
        qz_x = self.q_z(x)
        dkl = tf.reduce_mean(qz_x.kl_divergence(self.p_z))

        # Monte-Carlo estimate expected log likelihood
        z_samples = tf.reshape(qz_x.sample(sample_shape=self.num_mc_samples), [-1, self.dim_z])
        mean = tf.reshape(self.mean(z_samples), [self.num_mc_samples, -1] + list(self.dim_x))
        precision = tf.reshape(self.precision(z_samples), [self.num_mc_samples, -1] + list(self.dim_x))
        p_x_z = tfpd.Independent(tfpd.Normal(mean, precision ** -0.5), tf.rank(mean) - 2)
        ell = tf.reduce_mean(p_x_z.log_prob(x))

        # negative evidence lower bound
        loss = -(ell - dkl)

        return loss, mean, precision

    def predictive_distribution(self, *args):
        mean, precision = self.call(args[0], training=False)[1:] if len(args) == 1 else args
        permutation = tf.concat([tf.range(1, tf.rank(mean)), [0]], axis=0)
        mean = tf.transpose(mean, permutation)
        precision = tf.transpose(precision, permutation)
        p_x_x = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(logits=tf.ones(tf.shape(mean)[-1])),
            components_distribution=tfpd.Normal(loc=mean, scale=precision ** -0.5))
        return p_x_x

    def update_metrics(self, x, mean, precision):
        p_x_x = self.predictive_distribution(mean, precision)
        std_errors = (x - p_x_x.mean()) / p_x_x.stddev()
        prob_errors = tfpd.Normal(loc=tf.zeros_like(x), scale=tf.ones_like(x)).cdf(std_errors)
        predictor_values = pack_predictor_values(p_x_x.mean(), p_x_x.log_prob(x), prob_errors)
        self.compiled_metrics.update_state(y_true=x, y_pred=predictor_values)


if __name__ == '__main__':

    # enable background tiles on plots
    sns.set(color_codes=True)

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='which model to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--model', type=str, default='Normal', help='which model to use')
    parser.add_argument('--empirical_bayes', action='store_true', default=False, help='for Variational Gamma-Normal')
    parser.add_argument('--sq_err_scale', type=float, default=1.0, help='for Variational Gamma-Normal')
    parser.add_argument('--sparse', action='store_true', default=False, help='sparse toy data option')
    args = parser.parse_args()

    # load data
    ds_train, ds_valid = load_data('mnist', batch_size=args.batch_size)

    # pick the appropriate model
    plot_title = args.model
    if args.model == 'Normal':
        MODEL = NormalVAE
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(
        dim_x=(28, 28, 1),
        dim_z=10,
        encoder=encoder_dense,
        decoder=decoder_dense,
        num_mc_samples=20,
        empirical_bayes=args.empirical_bayes,  # for Variational Gamma-Normal
        sq_err_scale=args.sq_err_scale,  # for Variational Gamma-Normal
    )

    # build the model
    optimizer = tf.keras.optimizers.Adam(1e-4)
    mdl.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
        MeanLogLikelihood(),
        RootMeanSquaredError(),
        ExpectedCalibrationError(),
    ])

    # train model
    validation_freq = 500
    hist = mdl.fit(x=ds_train, validation_data=ds_valid, validation_freq=validation_freq, epochs=int(20e3), verbose=2)
