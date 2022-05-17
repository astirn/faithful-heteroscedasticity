import argparse

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp

from matplotlib import pyplot as plt
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

    def predict_step(self, data):
        x = self.parse_keras_inputs(data)

        return x, *self.call(x, training=False)[1:]

    def train_step(self, data):
        x = self.parse_keras_inputs(data)

        # optimization step
        with tf.GradientTape() as tape:
            loss, *params = self.call(x, training=True)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        # update metrics
        self.update_metrics(x, *params)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x = self.parse_keras_inputs(data)

        # update metrics
        self.update_metrics(x, *self.call(x, training=False)[1:])

        return {m.name: m.result() for m in self.metrics}


class NormalVAE(HeteroscedasticVariationalAutoencoder):

    def __init__(self, dim_x, dim_z, decoder, optimization, **kwargs):
        name = 'NormalVAE' + '-' + optimization
        HeteroscedasticVariationalAutoencoder.__init__(self, dim_x, dim_z, name=name, **kwargs)

        # save optimization method
        self.optimization = optimization

        # decoder networks
        arch = [128, 256, 512]
        self.mean = decoder(dim_z, arch, np.prod(dim_x), f_out=None, name='mu', **kwargs)
        self.precision = decoder(dim_z, arch, np.prod(dim_x), f_out='softplus', name='lambda', **kwargs)

    def call(self, x, **kwargs):

        # variational family and its KL-divergence
        qz_x = self.q_z(x, **kwargs)
        dkl = tf.reduce_mean(qz_x.kl_divergence(self.p_z))

        # Monte-Carlo estimate expected log likelihood
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)
        z_samples = tf.reshape(tf.transpose(z_samples, [1, 0, 2]), [-1, self.dim_z])
        mean = tf.reshape(self.mean(z_samples, **kwargs), [-1, self.num_mc_samples] + list(self.dim_x))
        precision = tf.reshape(self.precision(z_samples, **kwargs), [-1, self.num_mc_samples] + list(self.dim_x))
        if self.optimization == 'first-order':
            p_x_z = tfpd.Independent(tfpd.Normal(mean, precision ** -0.5), tf.rank(mean) - 2)
            ell = tf.reduce_mean(p_x_z.log_prob(x[:, None, ...]))
        elif self.optimization == 'second-order-mean':
            sq_error = (x[:, None, ...] - mean) ** 2
            sq_error = tf.reduce_sum(tf.reshape(sq_error, tf.concat([tf.shape(sq_error)[:2], [-1]], axis=0)), axis=-1)
            p_x_z = tfpd.Independent(tfpd.Normal(tf.stop_gradient(mean), precision ** -0.5), tf.rank(mean) - 2)
            ell = tf.reduce_mean(p_x_z.log_prob(x[:, None, ...]) - 0.5 * sq_error)
        else:
            raise NotImplementedError

        # negative evidence lower bound
        loss = -(ell - dkl)

        return loss, mean, precision

    def predictive_distribution(self, *args):
        mean, precision = self.call(args[0], training=False)[1:] if len(args) == 1 else args
        permutation = tf.concat([[0], tf.range(2, tf.rank(mean)), [1]], axis=0)
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


class GammaNormalVAE(HeteroscedasticVariationalAutoencoder):

    def __init__(self, dim_x, dim_z, decoder, **kwargs):
        name = 'GammaNormalVAE'
        HeteroscedasticVariationalAutoencoder.__init__(self, dim_x, dim_z, name=name, **kwargs)

        # precision prior
        # self.empirical_bayes = empirical_bayes
        # self.sq_err_scale = tf.Variable(sq_err_scale if isinstance(sq_err_scale, float) else 1.0, trainable=sq_err_scale == 'T', name='sq_err_scale')
        self.a = tf.Variable(3 * tf.ones(self.dim_x), trainable=False)
        self.b = tf.Variable(1e-3 * (3 - 1) * tf.ones(self.dim_x), trainable=False)

        # decoder networks
        arch = [128, 256, 512]
        self.mu = decoder(dim_z, arch, np.prod(dim_x), f_out=None, name='mu', **kwargs)
        self.alpha = decoder(dim_z, arch, np.prod(dim_x), f_out=lambda x: 1 + tf.nn.softplus(x), name='alpha', **kwargs)
        self.beta = decoder(dim_z, arch, np.prod(dim_x), f_out='softplus', name='beta', **kwargs)

    def call(self, x, **kwargs):

        # variational family and Monte-Carlo samples
        qz_x = self.q_z(x, **kwargs)
        z_samples = qz_x.sample(sample_shape=self.num_mc_samples)
        z_samples = tf.reshape(tf.transpose(z_samples, [1, 0, 2]), [-1, self.dim_z])
        alpha = tf.reshape(self.alpha(z_samples, **kwargs), [-1, self.num_mc_samples] + list(self.dim_x))
        beta = tf.reshape(self.beta(z_samples, **kwargs), [-1, self.num_mc_samples] + list(self.dim_x))
        qp_z = tfpd.Independent(tfpd.Gamma(alpha, beta), reinterpreted_batch_ndims=len(self.dim_x))

        # expected log likelihood
        mu = tf.reshape(self.mu(z_samples, **kwargs), [-1, self.num_mc_samples] + list(self.dim_x))
        expected_lambda = alpha / beta
        expected_ln_lambda = tf.math.digamma(alpha) - tf.math.log(beta)
        ell = 0.5 * (expected_ln_lambda - tf.math.log(2 * np.pi) - (x[:, None, ...] - mu) ** 2 * expected_lambda)
        ell = tf.reduce_mean(tf.reduce_sum(tf.reshape(ell, tf.concat([tf.shape(ell)[:2], [-1]], axis=0)), axis=-1))

        # precision prior
        p_lambda = tfpd.Independent(tfpd.Gamma(self.a, self.b), tf.rank(x) - 1)

        # KL divergences
        dkl_z = tf.reduce_mean(qz_x.kl_divergence(self.p_z))
        dkl_p = tf.reduce_mean(qp_z.kl_divergence(p_lambda))

        # negative evidence lower bound
        loss = -(ell - dkl_z - dkl_p)

        return loss, mu, alpha, beta

    @staticmethod
    def predictive_distribution(mu, alpha, beta):
        permutation = tf.concat([[0], tf.range(2, tf.rank(mu)), [1]], axis=0)
        df = tf.transpose(2 * alpha, permutation)
        mean = tf.transpose(mu, permutation)
        scale = tf.transpose(tf.sqrt(beta / alpha), permutation)
        p_x_x = tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(logits=tf.ones(tf.shape(mean)[-1])),
            components_distribution=tfpd.StudentT(df=df, loc=mean, scale=scale))
        return p_x_x

    def update_metrics(self, x, mu, alpha, beta):
        p_x_x = self.predictive_distribution(mu, alpha, beta)
        df = tf.reduce_mean(2 * alpha, axis=1)
        std_errors = tf.reduce_mean((x[:, None, ...] - mu) / tf.sqrt(beta / alpha), axis=1)
        prob_errors = tfpd.StudentT(df=df, loc=tf.zeros_like(x), scale=tf.ones_like(x)).cdf(std_errors)
        predictor_values = pack_predictor_values(p_x_x.mean(), p_x_x.log_prob(x), prob_errors)
        self.compiled_metrics.update_state(y_true=x, y_pred=predictor_values)


def plot_posterior_predictive_checks(x, p_x_x, title, num_samples=10, column_wise=False):

    # randomly select data subset
    i = np.random.choice(x.shape[0], num_samples, replace=False)

    # concatenate each randomly selected data and its PPC along the non-plotting axis
    axis = int(not column_wise)
    x_plot = np.concatenate([np.squeeze(xx) for xx in np.split(x[i], num_samples)], axis=axis)
    mean = np.concatenate([np.squeeze(xx) for xx in np.split(p_x_x.mean().numpy()[i], num_samples)], axis=axis)
    std = np.concatenate([np.squeeze(xx) for xx in np.split(p_x_x.stddev().numpy()[i], num_samples)], axis=axis)
    sample = np.concatenate([np.squeeze(xx) for xx in np.split(p_x_x.sample().numpy()[i], num_samples)], axis=axis)

    # plot results
    fig, ax = plt.subplots(1, 1)
    fig.suptitle(title)
    ppc = np.concatenate([x_plot, mean, std, sample], axis=int(column_wise))
    ax.imshow(ppc, vmin=0, vmax=1, cmap='gray_r')


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='which model to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--model', type=str, default='NormalVAE', help='which model to use')
    parser.add_argument('--optimization', type=str, default='first-order', help='for Normal VAE')
    parser.add_argument('--empirical_bayes', action='store_true', default=False, help='for Gamma-Normal VAE')
    parser.add_argument('--sq_err_scale', type=float, default=1.0, help='for Gamma-Normal VAE')
    args = parser.parse_args()

    # load data
    ds_train, ds_valid = load_data('mnist', batch_size=args.batch_size)

    # pick the appropriate model
    plot_title = args.model
    if args.model == 'NormalVAE':
        MODEL = NormalVAE
    if args.model == 'GammaNormalVAE':
        MODEL = GammaNormalVAE
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(
        dim_x=(28, 28, 1),
        dim_z=10,
        encoder=encoder_dense,
        decoder=decoder_dense,
        num_mc_samples=20,
        optimization=args.optimization,  # for Normal VAE
        empirical_bayes=args.empirical_bayes,  # for Gamma-Normal VAE
        sq_err_scale=args.sq_err_scale,  # for Gamma-Normal VAE
    )

    # build the model
    optimizer = tf.keras.optimizers.Adam(5e-5)
    mdl.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
        MeanLogLikelihood(),
        RootMeanSquaredError(),
        ExpectedCalibrationError(),
    ])

    # train model
    validation_freq = 1
    hist = mdl.fit(x=ds_train, validation_data=ds_valid, validation_freq=validation_freq, epochs=int(20e3), verbose=0,
                   callbacks=[RegressionCallback(validation_freq=validation_freq, early_stop_patience=6)])

    # posterior predictive parameters for validation data
    x_valid, *params = mdl.predict(ds_valid)

    # plot posterior predictive checks
    plot_posterior_predictive_checks(x_valid, mdl.predictive_distribution(*params), title=mdl.name)
    plt.show()
