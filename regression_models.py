import argparse

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC
from matplotlib import pyplot as plt
from tensorflow_probability import distributions as tfpd

from callbacks import RegressionCallback
from regression_data import generate_toy_data


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

    def __init__(self, optimization='first-order', y_mean=0.0, y_var=1.0, **kwargs):
        tf.keras.Model.__init__(self, name=kwargs.get('name'))

        # save optimization method
        self.optimization = optimization

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

    def first_order_gradients(self, data):

        # take necessary gradients
        with tf.GradientTape() as tape:
            mean, precision = self.call(data, training=True)
            py_x = tfpd.MultivariateNormalDiag(loc=mean, scale_diag=precision ** -0.5)
            ll = py_x.log_prob(self.whiten_targets(data['y']))
            loss = tf.reduce_sum(tf.reduce_mean(-ll, axis=-1))
        gradients = tape.gradient(loss, self.trainable_variables)

        return gradients

    def second_order_gradients_mean(self, data):

        # take necessary gradients
        dim_batch = tf.cast(tf.shape(data['x'])[0], tf.float32)
        with tf.GradientTape(persistent=self.run_eagerly) as tape:
            mean, precision = self.call(data, training=True)
            py_x = tfpd.MultivariateNormalDiag(loc=tf.stop_gradient(mean), scale_diag=precision ** -0.5)
            y = self.whiten_targets(data['y'])
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

        return gradients

    def second_order_gradients_diag(self, data, f_mean, f_precision, diag):
        
        # take necessary gradients
        dim_batch = tf.cast(tf.shape(data['x'])[0], tf.float32)
        trainable_variables = f_mean.trainable_variables + f_precision.trainable_variables
        with tf.GradientTape(persistent=True) as tape2:
            with tf.GradientTape(persistent=True) as tape1:
                mean, precision = f_mean.call(data, training=True), f_precision.call(data, training=True)
                mean_precision = tf.stack([mean, precision], axis=-1)
                py_x = tfpd.MultivariateNormalDiag(loc=mean, scale_diag=precision ** -0.5)
                loss = tf.reduce_sum(tf.reduce_mean(-py_x.log_prob(self.whiten_targets(data['y'])), axis=-1))
            dl_dm = tape1.gradient(loss, mean)
            dl_dp = tape1.gradient(loss, precision)
            dmp_dnet = tape1.jacobian(mean_precision, trainable_variables)
        d2nll_dm2 = tape2.gradient(dl_dm, mean) * dim_batch
        d2nll_dp2 = tape2.gradient(dl_dp, precision) * dim_batch
        tf.assert_greater(d2nll_dp2, 0.0)
        d2nll_dp2 = tf.clip_by_value(d2nll_dp2, 1e-3, np.inf)

        # apply second order information
        if diag:
            dl_dmv = tf.stack([dl_dm / d2nll_dm2, dl_dp / d2nll_dp2], axis=-1)
        else:
            d2nll_dmdp = tape2.gradient(dl_dm, precision)
            H = tf.reshape(tf.concat([d2nll_dm2, d2nll_dmdp, d2nll_dmdp, d2nll_dp2], axis=-1), [-1, 2, 2])
            dl_dmv = tf.transpose(tf.linalg.solve(H, tf.stack([dl_dm, dl_dp], axis=-2)), [0, 2, 1])
        gradients = [tf.tensordot(dl_dmv, d, axes=[[0, 1, 2], [0, 1, 2]]) for d in dmp_dnet]

        return gradients, trainable_variables

    def train_step(self, data):
        if self.optimization == 'first-order':
            self.optimizer.apply_gradients(zip(self.first_order_gradients(data), self.trainable_variables))
        elif self.optimization == 'second-order-mean':
            self.optimizer.apply_gradients(zip(self.second_order_gradients_mean(data), self.trainable_variables))
        elif self.optimization in {'second-order-diag', 'second-order-full'}:
            diag = 'diag' in self.optimization
            if isinstance(self.f_mean, (list, tuple)) and isinstance(self.f_precision, (list, tuple)):
                for f_mean, f_precision in zip(self.f_mean, self.f_precision):
                    gradients, network_params = self.second_order_gradients_diag(data, f_mean, f_precision, diag)
                    self.optimizer.apply_gradients(zip(gradients, network_params))
            else:
                gradients, network_params = self.second_order_gradients_diag(data, self.f_mean, self.f_precision, diag)
                self.optimizer.apply_gradients(zip(gradients, network_params))
        else:
            raise NotImplementedError

        # update metrics
        mean, variance = self.predictive_central_moments(data['x'])
        self.compiled_metrics.update_state(data['y'], mean)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        mean, variance = self.predictive_central_moments(data['x'])
        self.compiled_metrics.update_state(data['y'], mean)

        return {m.name: m.result() for m in self.metrics}


class Normal(HeteroscedasticRegression, ABC):

    def __init__(self, dim_x, dim_y, **kwargs):
        HeteroscedasticRegression.__init__(self, name='Normal', **kwargs)

        # define parameter networks
        self.f_mean = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu', **kwargs)
        self.f_precision = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='lambda', **kwargs)

    def call(self, inputs, **kwargs):
        return self.f_mean(inputs['x'], **kwargs), self.f_precision(inputs['x'], **kwargs)

    def predictive_central_moments(self, x):
        mean = self.de_whiten_mean(self.f_mean(x, training=False))
        variance = self.de_whiten_precision(self.f_precision(x, training=False)) ** -1

        return mean, variance

    def predictive_distribution(self, x):
        mean = self.de_whiten_mean(self.f_mean(x, training=False))
        stddev = self.de_whiten_precision(self.f_precision(x, training=False)) ** -0.5
        px_y = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=stddev)

        return px_y


class MonteCarloDropout(HeteroscedasticRegression, ABC):

    def __init__(self, dim_x, dim_y, num_mc_samples, **kwargs):
        HeteroscedasticRegression.__init__(self, name='MonteCarloDropout', **kwargs)

        # save configuration
        self.num_mc_samples = num_mc_samples

        # define parameter networks
        self.mean = neural_network(d_in=dim_x, d_out=dim_y, f_out=None, rate=0.1, name='mu', **kwargs)
        self.precision = neural_network(d_in=dim_x, d_out=dim_y, f_out='softplus', rate=0.1, name='lambda', **kwargs)

    def call(self, inputs, **kwargs):
        return self.mean(inputs['x'], **kwargs), self.precision(inputs['x'], **kwargs)

    def predictive_central_moments(self, x):
        means = tf.stack([self.mean(x, training=True) for _ in range(self.num_mc_samples)], axis=0)
        variances = tf.stack([self.precision(x, training=True) ** -1 for _ in range(self.num_mc_samples)], axis=0)
        predictive_mean = tf.reduce_mean(means, axis=0)
        predictive_variance = tf.reduce_mean(means ** 2 + variances, axis=0) - tf.reduce_mean(means, axis=0) ** 2

        return self.de_whiten_mean(predictive_mean), self.de_whiten_variance(predictive_variance)

    def predictive_distribution(self, x):
        raise NotImplementedError


class DeepEnsemble(HeteroscedasticRegression, ABC):

    def __init__(self, dim_x, dim_y, num_ensembles, **kwargs):
        HeteroscedasticRegression.__init__(self, name='DeepEnsemble', **kwargs)

        # define parameter networks
        self.f_mean, self.f_precision = [], []
        for i in range(num_ensembles):
            s = str(i + 1)
            self.f_mean += [param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu_' + s, **kwargs)]
            self.f_precision += [param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='lambda_' + s, **kwargs)]

    def call(self, inputs, **kwargs):
        means = tf.stack([mean(inputs['x'], **kwargs) for mean in self.f_mean], axis=0)
        precisions = tf.stack([precision(inputs['x'], **kwargs) for precision in self.f_precision], axis=0)
        return means, precisions

    def predictive_central_moments(self, x):
        means = tf.stack([mean(x, training=False) for mean in self.f_mean], axis=0)
        variances = tf.stack([precision(x, training=False) ** -1 for precision in self.f_precision], axis=0)
        predictive_mean = tf.reduce_mean(means, axis=0)
        predictive_variance = tf.reduce_mean(means ** 2 + variances, axis=0) - tf.reduce_mean(means, axis=0) ** 2

        return self.de_whiten_mean(predictive_mean), self.de_whiten_variance(predictive_variance)

    def predictive_distribution(self, x):
        raise NotImplementedError


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
    parser.add_argument('--model', type=str, default='Normal', help='{Normal, MC-dropout, Ensemble}')
    parser.add_argument('--optimization', type=str, default='first-order', help='how to compute gradients')
    parser.add_argument('--sparse', action='store_true', default=False, help='sparse toy data option')
    args = parser.parse_args()

    # load data
    x_train, y_train, _, _, _ = generate_toy_data(sparse=bool(args.sparse))
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])
    x_valid, y_valid, x_eval, true_mean, true_std = generate_toy_data(sparse=bool(args.sparse))
    ds_valid = tf.data.Dataset.from_tensor_slices({'x': x_valid, 'y': y_valid}).batch(x_valid.shape[0])

    # pick the appropriate model
    plot_title = args.model
    if args.model == 'Normal':
        MODEL = Normal
    elif args.model == 'MonteCarloDropout':
        MODEL = MonteCarloDropout
    elif args.model == 'DeepEnsemble':
        MODEL = DeepEnsemble
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(
        dim_x=x_train.shape[1],
        dim_y=y_train.shape[1],
        optimization=args.optimization,
        y_mean=tf.constant([y_train.mean()], dtype=tf.float32),
        y_var=tf.constant([y_train.var()], dtype=tf.float32),
        num_mc_samples=20,
        num_ensembles=10,
    )

    # build the model
    optimizer = tf.keras.optimizers.Adam(5e-3)
    mdl.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train model
    hist = mdl.fit(x=ds_train, validation_data=ds_valid, epochs=int(20e3), verbose=0, callbacks=[
        RegressionCallback(validation_freq=500, early_stop_patience=6),
    ])

    # evaluate predictive model
    mdl.num_mc_samples = 2000
    mdl_mean, mdl_var = mdl.predictive_central_moments(x_eval)
    mdl_mean, mdl_std = mdl_mean.numpy(), mdl_var.numpy() ** 0.5

    # plot results for toy data
    fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, args.model + ' ' + mdl.optimization)
    plt.show()
