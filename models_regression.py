import argparse
import numpy as np
import seaborn as sns
import tensorflow as tf
from abc import ABC
from callbacks import RegressionCallback
from data_regression import generate_toy_data
from matplotlib import pyplot as plt
from metrics import pack_predictor_values, MeanLogLikelihood, RootMeanSquaredError, ExpectedCalibrationError
from tensorflow_probability import distributions as tfpd

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


class Regression(tf.keras.Model):

    def __init__(self, **kwargs):
        tf.keras.Model.__init__(self, name=kwargs['name'])

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
        self.update_metrics(y, **params)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = self.parse_keras_inputs(data)

        # update metrics
        self.update_metrics(y, **self.call(x, training=True))

        return {m.name: m.result() for m in self.metrics}


class HomoscedasticNormal(Regression, ABC):

    def __init__(self, dim_x, dim_y, **kwargs):
        Regression.__init__(self, name='HomoscedasticNormal', **kwargs)

        # parameter networks
        self.f_mean = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)

    def call(self, x, **kwargs):
        return {'mean': self.f_mean(x, **kwargs)}

    def predictive_distribution(self, *, x=None, mean=None):
        if mean is None:
            assert x is not None
            mean, = self.call(x, training=False).values()
        return tfpd.Normal(loc=mean, scale=1.0)

    def update_metrics(self, y, mean):
        py_x = self.predictive_distribution(mean=mean)
        prob_errors = tfpd.Normal(0, 1).cdf((y - py_x.mean()) / py_x.stddev())
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(loc=params['mean'], scale=1.0), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class HeteroscedasticNormal(Regression, ABC):

    def __init__(self, dim_x, dim_y, **kwargs):
        Regression.__init__(self, name=kwargs.pop('name', 'HeteroscedasticNormal'), **kwargs)

        # parameter networks
        self.f_mean = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
        self.f_scale = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)

    def call(self, x, **kwargs):
        return {'mean': self.f_mean(x, **kwargs), 'std': self.f_scale(x, **kwargs)}

    def predictive_distribution(self, *, x=None, mean=None, std=None):
        if mean is None or std is None:
            assert x is not None
            mean, std = self.call(x, training=False).values()
        return tfpd.Normal(loc=mean, scale=std)

    def update_metrics(self, y, mean, std):
        py_x = self.predictive_distribution(mean=mean, std=std)
        prob_errors = tfpd.Normal(0, 1).cdf((y - py_x.mean()) / py_x.stddev())
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(*params.values()), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class FaithfulHeteroscedasticNormal(HeteroscedasticNormal, ABC):

    def __init__(self, dim_x, dim_y, **kwargs):
        HeteroscedasticNormal.__init__(self, dim_x, dim_y, name='FaithfulHeteroscedasticNormal', **kwargs)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_loc = tfpd.Independent(tfpd.Normal(loc=params['mean'], scale=1.0),
                                      reinterpreted_batch_ndims=1)
            py_std = tfpd.Independent(tfpd.Normal(loc=tf.stop_gradient(params['mean']), scale=params['std']),
                                      reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_loc.log_prob(y) + py_std.log_prob(y))
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class Student(Regression):

    def __init__(self, dim_x, dim_y, **kwargs):
        Regression.__init__(self, name='Student', **kwargs)

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
        return tfpd.StudentT(df=2 * alpha, loc=loc, scale=scale)

    def update_metrics(self, y, mu, alpha, beta):
        py_x = self.predictive_distribution(mu, alpha, beta)
        scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
        prob_errors = tfpd.StudentT(df=2 * alpha, loc=0, scale=1).cdf((y - py_x.mean()) / scale)
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):

        with tf.GradientTape() as tape:

            # amortized parameter networks
            mu, alpha, beta = self.call(x, training=True)

            # minimize negative log likelihood
            py_x = tfpd.StudentT(df=2 * alpha, loc=mu, scale=tf.sqrt(beta / alpha))
            ll = tf.reduce_sum(py_x.log_prob(self.whiten_targets(y)), axis=-1)
            loss = tf.reduce_mean(-ll)

        # update model parameters
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        return mu, alpha, beta


class VariationalGammaNormal(Regression):

    def __init__(self, dim_x, dim_y, empirical_bayes, sq_err_scale=None, **kwargs):
        assert not empirical_bayes or sq_err_scale is not None
        name = 'VariationalGammaNormal' + ('-EB-{:.2f}'.format(sq_err_scale) if empirical_bayes else '')
        Regression.__init__(self, name=name, **kwargs)

        # precision prior
        self.empirical_bayes = empirical_bayes
        self.sq_err_scale = sq_err_scale
        self.p_lambda = tfpd.Independent(tfpd.Gamma([2.0] * dim_y, [1.0] * dim_y), 1)

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
        return tfpd.StudentT(df=2 * alpha, loc=loc, scale=scale)

    def update_metrics(self, y, mu, alpha, beta):
        py_x = self.predictive_distribution(mu, alpha, beta)
        scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
        prob_errors = tfpd.StudentT(df=2 * alpha, loc=0, scale=1).cdf((y - py_x.mean()) / scale)
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):

        # empirical bayes prior
        if self.empirical_bayes:
            sq_errors = (self.whiten_targets(y) - self.mu(x)) ** 2
            a = tf.reduce_mean(sq_errors, axis=0) ** 2 / tf.math.reduce_variance(sq_errors, axis=0) + 2
            b = (a - 1) * tf.reduce_mean(sq_errors, axis=0) / self.sq_err_scale
            p_lambda = tfpd.Independent(tfpd.Gamma(a, b), 1)

        # standard prior
        else:
            p_lambda = self.p_lambda

        with tf.GradientTape() as tape:

            # amortized parameter networks
            mu, alpha, beta = self.call(x, training=True)

            # variational family
            qp = tfpd.Independent(tfpd.Gamma(alpha, beta), reinterpreted_batch_ndims=1)

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
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--model', type=str, default='HeteroscedasticNormal', help='which model to use')
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
    if args.model == 'HomoscedasticNormal':
        MODEL = HomoscedasticNormal
    elif args.model == 'HeteroscedasticNormal':
        MODEL = HeteroscedasticNormal
    elif args.model == 'FaithfulHeteroscedasticNormal':
        MODEL = FaithfulHeteroscedasticNormal
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(
        dim_x=x_train.shape[1],
        dim_y=y_train.shape[1],
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
        ExpectedCalibrationError(),
    ])

    # train model
    validation_freq = 500
    hist = mdl.fit(x=ds_train, validation_data=ds_valid, validation_freq=validation_freq, epochs=int(30e3), verbose=0,
                   callbacks=[RegressionCallback(validation_freq=500, early_stop_patience=0)])

    # evaluate predictive model
    mdl.num_mc_samples = 2000
    p_y_x = mdl.predictive_distribution(x=x_eval)
    mdl_mean, mdl_std = p_y_x.mean().numpy(), p_y_x.stddev().numpy()

    # plot results for toy data
    fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, args.model)
    plt.show()
