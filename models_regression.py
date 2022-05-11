import argparse

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC
from matplotlib import pyplot as plt
from tensorflow_probability import distributions as tfpd

from callbacks import RegressionCallback
from metrics import pack_predictor_values, MeanLogLikelihood, RootMeanSquaredError, ExpectationCalibrationError
from data_regression import generate_toy_data


# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def param_net(d_in, d_out, n_hidden=1, d_hidden=50, f_hidden='elu', rate=0.0, f_out=None, name=None, **kwargs):
    assert isinstance(d_in, int) and d_in > 0
    assert isinstance(d_hidden, int) and d_hidden > 0
    assert isinstance(d_out, int) and d_out > 0
    nn = tf.keras.Sequential(name=name)
    nn.add(tf.keras.layers.InputLayer(d_in))
    for _ in range(n_hidden):
        nn.add(tf.keras.layers.Dense(d_hidden, f_hidden))
        nn.add(tf.keras.layers.Dropout(rate))
    nn.add(tf.keras.layers.Dense(d_out, f_out))
    return nn


class HeteroscedasticRegression(tf.keras.Model):

    def __init__(self, y_mean, y_var, **kwargs):
        tf.keras.Model.__init__(self, name=kwargs['name'])

        # save target scaling
        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)

    def whiten_targets(self, y):
        return (y - self.y_mean) / self.y_std

    def de_whiten_mean(self, mu):
        return mu * self.y_std + self.y_mean

    def de_whiten_stddev(self, sigma):
        return sigma * self.y_std

    def de_whiten_variance(self, variance):
        return variance * self.y_var

    def de_whiten_precision(self, precision):
        return precision / self.y_var

    def de_whiten_log_precision(self, log_precision):
        return log_precision - tf.math.log(self.y_var)

    @staticmethod
    def parse_keras_inputs(data):
        if isinstance(data, dict):
            x, y = data['x'], data['y']
        elif isinstance(data, tuple):
            x, y = data
        else:
            raise NotImplementedError
        return x, y

    def train_step(self, data):
        x, y = self.parse_keras_inputs(data)

        # optimization step
        params = self.optimization_step(x, y)

        # update metrics
        self.update_metrics(y, *params)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = self.parse_keras_inputs(data)

        # update metrics
        self.update_metrics(y, *self.call(x, training=True))

        return {m.name: m.result() for m in self.metrics}


class Normal(HeteroscedasticRegression, ABC):

    def __init__(self, dim_x, dim_y, optimization, **kwargs):
        HeteroscedasticRegression.__init__(self, name='Normal' + '-' + optimization, **kwargs)

        # save optimization method
        self.optimization = optimization

        # parameter networks
        self.f_mean = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu', **kwargs)
        self.f_precision = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='lambda', **kwargs)

    def call(self, x, **kwargs):
        return self.f_mean(x, **kwargs), self.f_precision(x, **kwargs)

    def predictive_distribution(self, *args):
        mean, precision = self.call(args[0], training=False) if len(args) == 1 else args
        mean = self.de_whiten_mean(mean)
        stddev = self.de_whiten_precision(precision) ** -0.5
        return tfp.distributions.Normal(loc=mean, scale=stddev)

    def update_metrics(self, y, mean, precision):
        py_x = self.predictive_distribution(mean, precision)
        prob_errors = tfp.distributions.Normal(0, 1).cdf((y - py_x.mean()) / py_x.stddev())
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def first_order_gradients(self, x, y):

        # take necessary gradients
        with tf.GradientTape() as tape:
            mean, precision = self.call(x, training=True)
            py_x = tfpd.MultivariateNormalDiag(loc=mean, scale_diag=precision ** -0.5)
            ll = py_x.log_prob(self.whiten_targets(y))
            loss = tf.reduce_sum(tf.reduce_mean(-ll, axis=-1))
        gradients = tape.gradient(loss, self.trainable_variables)

        return gradients, mean, precision

    def second_order_gradients_mean(self, x, y):

        # take necessary gradients
        dim_batch = tf.cast(tf.shape(x)[0], tf.float32)
        with tf.GradientTape(persistent=self.run_eagerly) as tape:
            mean, precision = self.call(x, training=True)
            py_x = tfpd.MultivariateNormalDiag(loc=tf.stop_gradient(mean), scale_diag=precision ** -0.5)
            y = self.whiten_targets(y)
            error = (y - mean)
            loss = tf.reduce_sum(tf.reduce_mean(0.5 * tf.reduce_sum(error ** 2, axis=-1) - py_x.log_prob(y), axis=-1))
        gradients = tape.gradient(loss, self.trainable_variables)

        # if we are debugging, make sure our gradient assumptions hold
        if self.run_eagerly:
            dl_dm_automatic = tape.gradient(loss, mean)
            dl_dm_expected = -error / dim_batch
            tf.assert_less(tf.abs(dl_dm_automatic - dl_dm_expected), 1e-5)
            dl_dp_automatic = tape.gradient(loss, precision)
            dl_dp_expected = 0.5 * (error ** 2 - precision ** -1) / dim_batch
            tf.assert_less(tf.abs(dl_dp_automatic - dl_dp_expected), 1e-5)

        return gradients, mean, precision

    def optimization_step(self, x, y):

        if self.optimization == 'first-order':
            gradients, mean, precision = self.first_order_gradients(x, y)
        elif self.optimization == 'second-order-mean':
            gradients, mean, precision = self.second_order_gradients_mean(x, y)
        else:
            raise NotImplementedError

        # update model parameters
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return mean, precision


class Student(HeteroscedasticRegression):

    def __init__(self, dim_x, dim_y, **kwargs):
        HeteroscedasticRegression.__init__(self, name='Student', **kwargs)

        # parameter networks
        self.mu = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu', **kwargs)
        self.alpha = param_net(d_in=dim_x, d_out=dim_y, f_out=lambda x: 1 + tf.nn.softplus(x), name='alpha', **kwargs)
        self.beta = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='beta', **kwargs)

    def call(self, x, **kwargs):
        return self.mu(x, **kwargs), self.alpha(x, **kwargs), self.beta(x, **kwargs)

    def predictive_distribution(self, *args):
        mu, alpha, beta = self.call(args[0], training=False) if len(args) == 1 else args
        loc = self.de_whiten_mean(mu)
        scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
        return tfp.distributions.StudentT(df=2 * alpha, loc=loc, scale=scale)

    def update_metrics(self, y, mu, alpha, beta):
        py_x = self.predictive_distribution(mu, alpha, beta)
        prob_errors = tfp.distributions.StudentT(df=2 * alpha, loc=0, scale=1).cdf((y - py_x.mean()) / py_x.stddev())
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):

        with tf.GradientTape() as tape:

            # amortized parameter networks
            mu, alpha, beta = self.call(x, training=True)

            # minimize negative log likelihood
            py_x = tfp.distributions.StudentT(df=2 * alpha, loc=mu, scale=tf.sqrt(beta / alpha))
            ll = tf.reduce_sum(py_x.log_prob(self.whiten_targets(y)), axis=-1)
            loss = tf.reduce_mean(-ll)

        # update model parameters
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        return mu, alpha, beta


class VariationalGammaNormal(HeteroscedasticRegression):

    def __init__(self, dim_x, dim_y, empirical_bayes, sq_err_scale=None, **kwargs):
        assert not empirical_bayes or sq_err_scale is not None
        name = 'VariationalGammaNormal' + ('-EB-{:.2f}'.format(sq_err_scale) if empirical_bayes else '')
        HeteroscedasticRegression.__init__(self, name=name, **kwargs)

        # precision prior
        self.empirical_bayes = empirical_bayes
        self.sq_err_scale = sq_err_scale
        self.p_lambda = tfp.distributions.Independent(tfp.distributions.Gamma([2.0] * dim_y, [1.0] * dim_y), 1)

        # parameter networks
        self.mu = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu', **kwargs)
        self.alpha = param_net(d_in=dim_x, d_out=dim_y, f_out=lambda x: 1 + tf.nn.softplus(x), name='alpha', **kwargs)
        self.beta = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='beta', **kwargs)

    def call(self, x, **kwargs):
        return self.mu(x, **kwargs), self.alpha(x, **kwargs), self.beta(x, **kwargs)

    def predictive_distribution(self, *args):
        mu, alpha, beta = self.call(args[0], training=False) if len(args) == 1 else args
        loc = self.de_whiten_mean(mu)
        scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
        return tfp.distributions.StudentT(df=2 * alpha, loc=loc, scale=scale)

    def update_metrics(self, y, mu, alpha, beta):
        py_x = self.predictive_distribution(mu, alpha, beta)
        prob_errors = tfp.distributions.StudentT(df=2 * alpha, loc=0, scale=1).cdf((y - py_x.mean()) / py_x.stddev())
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):

        # empirical bayes prior
        if self.empirical_bayes:
            sq_errors = (self.whiten_targets(y) - self.mu(x)) ** 2
            a = tf.reduce_mean(sq_errors, axis=0) ** 2 / tf.math.reduce_variance(sq_errors, axis=0) + 2
            b = (a - 1) * tf.reduce_mean(sq_errors, axis=0) / self.sq_err_scale
            p_lambda = tfp.distributions.Independent(tfp.distributions.Gamma(a, b), 1)

        # standard prior
        else:
            p_lambda = self.p_lambda

        with tf.GradientTape() as tape:

            # amortized parameter networks
            mu, alpha, beta = self.call(x, training=True)

            # variational family
            qp = tfp.distributions.Independent(tfp.distributions.Gamma(alpha, beta), reinterpreted_batch_ndims=1)

            # use negative evidence lower bound as minimization objective
            y = self.whiten_targets(y)
            expected_lambda = alpha / beta
            expected_ln_lambda = tf.math.digamma(alpha) - tf.math.log(beta)
            ell = 0.5 * (expected_ln_lambda - tf.math.log(2 * np.pi) - (y - mu) ** 2 * expected_lambda)
            loss = -tf.reduce_mean(tf.reduce_sum(ell, axis=-1) - qp.kl_divergence(p_lambda))

        # update model parameters
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        return mu, alpha, beta


def fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, title):
    # squeeze everything
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_eval = np.squeeze(x_eval)
    true_mean = np.squeeze(true_mean)
    true_std = np.squeeze(true_std)
    mdl_mean = np.squeeze(mdl_mean)
    mdl_std = np.squeeze(mdl_std)

    # get a new figure
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)

    # plot the data
    sns.scatterplot(x_train, y_train, ax=ax[0])

    # plot the true mean and standard deviation
    ax[0].plot(x_eval, true_mean, '--k')
    ax[0].plot(x_eval, true_mean + true_std, ':k')
    ax[0].plot(x_eval, true_mean - true_std, ':k')

    # plot the model's mean and standard deviation
    l = ax[0].plot(x_eval, mdl_mean)[0]
    ax[0].fill_between(x_eval[:, ], mdl_mean - mdl_std, mdl_mean + mdl_std, color=l.get_color(), alpha=0.5)
    ax[0].plot(x_eval, true_mean, '--k')

    # clean it up
    ax[0].set_xlim([0, 10])
    ax[0].set_ylim([-12.5, 12.5])
    ax[0].set_ylabel('y')

    # plot the std
    ax[1].plot(x_eval, mdl_std, label='predicted')
    ax[1].plot(x_eval, true_std, '--k', label='truth')
    ax[1].set_xlim([0, 10])
    ax[1].set_ylim([0, 6])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('std(y|x)')
    plt.legend()

    return fig


if __name__ == '__main__':

    # enable background tiles on plots
    sns.set(color_codes=True)

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--debug', action='store_true', default=False, help='sparse toy data option')
    parser.add_argument('--model', type=str, default='Normal', help='which model to use')
    parser.add_argument('--optimization', type=str, default='first-order', help='for Normal, MC-Dropout, DeepEnsemble')
    parser.add_argument('--empirical_bayes', action='store_true', default=False, help='for Variational Gamma-Normal')
    parser.add_argument('--sq_err_scale', type=float, default=1.0, help='for Variational Gamma-Normal')
    parser.add_argument('--sparse', action='store_true', default=False, help='sparse toy data option')
    args = parser.parse_args()

    # load data
    x_train, y_train, _, _, _ = generate_toy_data(num_samples=500, sparse=bool(args.sparse))
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])
    x_valid, y_valid, x_eval, true_mean, true_std = generate_toy_data(sparse=bool(args.sparse))
    ds_valid = tf.data.Dataset.from_tensor_slices({'x': x_valid, 'y': y_valid}).batch(x_valid.shape[0])

    # pick the appropriate model
    plot_title = args.model
    if args.model == 'Normal':
        MODEL = Normal
    elif args.model == 'Student':
        MODEL = Student
    elif args.model == 'VariationalGammaNormal':
        MODEL = VariationalGammaNormal
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(
        dim_x=x_train.shape[1],
        dim_y=y_train.shape[1],
        y_mean=tf.constant([y_train.mean()], dtype=tf.float32),
        y_var=tf.constant([y_train.var()], dtype=tf.float32),
        optimization=args.optimization,  # for Normal
        num_mc_samples=20,  # for MC-Dropout
        empirical_bayes=args.empirical_bayes,  # for Variational Gamma-Normal
        sq_err_scale=args.sq_err_scale,  # for Variational Gamma-Normal
    )

    # build the model
    optimizer = tf.keras.optimizers.Adam(5e-3)
    mdl.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
        MeanLogLikelihood(),
        RootMeanSquaredError(),
        ExpectationCalibrationError(),
    ])

    # train model
    validation_freq = 500
    hist = mdl.fit(x=ds_train, validation_data=ds_valid, validation_freq=validation_freq, epochs=int(20e3), verbose=0,
                   callbacks=[RegressionCallback(validation_freq=500, early_stop_patience=6)])

    # evaluate predictive model
    mdl.num_mc_samples = 2000
    p_y_x = mdl.predictive_distribution(x_eval)
    mdl_mean, mdl_std = p_y_x.mean().numpy(), p_y_x.stddev().numpy()

    # plot results for toy data
    fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, args.model)
    plt.show()
