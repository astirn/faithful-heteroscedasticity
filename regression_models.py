import argparse

import numpy as np
import seaborn as sns
import tensorflow as tf
import tensorflow_probability as tfp

from abc import ABC
from matplotlib import pyplot as plt

from callbacks import PretrainCallback, RegressionCallback
from regression_data import generate_toy_data


# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def neural_network(d_in, d_out, n_hidden=2, d_hidden=50, f_hidden='elu', f_out=None, name=None, **kwargs):
    assert isinstance(d_in, int) and d_in > 0
    assert isinstance(d_hidden, int) and d_hidden > 0
    assert isinstance(d_out, int) and d_out > 0
    nn = tf.keras.Sequential(name=name)
    nn.add(tf.keras.layers.InputLayer(d_in))
    for _ in range(n_hidden):
        nn.add(tf.keras.layers.Dense(d_hidden, f_hidden))
    nn.add(tf.keras.layers.Dense(d_out, f_out))
    return nn


class HeteroscedasticRegression(tf.keras.Model):

    def __init__(self, y_mean=0.0, y_var=1.0, **kwargs):
        tf.keras.Model.__init__(self, **kwargs)

        self.y_mean = tf.constant(y_mean, dtype=tf.float32)
        self.y_var = tf.constant(y_var, dtype=tf.float32)
        self.y_std = tf.sqrt(self.y_var)

    def whiten_targets(self, y):
        return (y - self.y_mean) / self.y_std

    def de_whiten_mean(self, mu):
        return mu * self.y_std + self.y_mean

    def de_whiten_stddev(self, sigma):
        return sigma * self.y_std

    def de_whiten_precision(self, precision):
        return precision / self.y_var

    def de_whiten_log_precision(self, log_precision):
        return log_precision - tf.math.log(self.y_var)

    def train_step(self, data):
        with tf.GradientTape() as tape:
            mean, var = self.call(data, training=True)
            py_x = tfp.distributions.MultivariateNormalDiag(loc=mean, scale_diag=var ** 0.5)
            ll = py_x.log_prob(self.whiten_targets(data['y']))
            loss = tf.reduce_mean(-ll)

        # update model parameters
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # update metrics
        self.compiled_metrics.update_state(data['y'], self.de_whiten_mean(mean))

        return {m.name: m.result() for m in self.metrics}


class Normal(HeteroscedasticRegression, ABC):

    def __init__(self, dim_x, dim_y, y_mean=0.0, y_var=1.0, **kwargs):
        HeteroscedasticRegression.__init__(self, y_mean, y_var, name='Normal')

        # define parameter networks
        self.mean = neural_network(d_in=dim_x, d_out=dim_y, f_out=None, name='mean', **kwargs)
        self.variance = neural_network(d_in=dim_x, d_out=dim_y, f_out='softplus', name='variance', **kwargs)

    def call(self, inputs, **kwargs):
        return self.mean(inputs['x'], **kwargs), self.variance(inputs['x'], **kwargs)

    def predictive_moments_and_samples(self, x):
        p_x_y = tfp.distributions.MultivariateNormalDiag(loc=self.de_whiten_mean(self.mean(x)),
                                                         scale_diag=self.de_whiten_stddev(self.variance(x) ** 0.5))
        return p_x_y.mean().numpy(), p_x_y.stddev().numpy(), p_x_y.sample().numpy()


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
    parser.add_argument('--model', type=str, default='Normal', help='algorithm')
    parser.add_argument('--sparse', action='store_true', default=False, help='sparse toy data option')
    args = parser.parse_args()

    # load data
    x_train, y_train, x_eval, true_mean, true_std = generate_toy_data(sparse=bool(args.sparse))
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train, 'y': y_train}).batch(x_train.shape[0])

    # pick the appropriate model
    plot_title = args.model
    if args.model == 'Normal':
        MODEL = Normal
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(dim_x=x_train.shape[1],
                dim_y=y_train.shape[1],
                y_mean=tf.constant([y_train.mean()], dtype=tf.float32),
                y_var=tf.constant([y_train.var()], dtype=tf.float32),
                num_mc_samples=50
                )

    # build the model
    optimizer = tf.keras.optimizers.Adam(5e-3)
    mdl.compile(optimizer=optimizer, run_eagerly=False, metrics=[tf.keras.metrics.RootMeanSquaredError()])

    # train model
    num_epochs = int(15e3)
    hist = mdl.fit(x=ds_train, epochs=num_epochs, verbose=0, callbacks=[
        RegressionCallback(num_epochs),
        PretrainCallback(num_epochs)])

    # evaluate predictive model with increased Monte-Carlo samples (if sampling is used by the particular model)
    mdl.num_mc_samples = 2000
    mdl_mean, mdl_std, mdl_samples = mdl.predictive_moments_and_samples(x_eval)

    # plot results for toy data
    fancy_plot(x_train, y_train, x_eval, true_mean, true_std, mdl_mean, mdl_std, args.model)
    plt.show()
