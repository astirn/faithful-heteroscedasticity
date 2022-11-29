import argparse
import os

import numpy as np
import tensorflow as tf

from abc import ABC
from callbacks import RegressionCallback
from datasets import generate_toy_data
from matplotlib import pyplot as plt
from metrics import pack_predictor_values, MeanLogLikelihood, RootMeanSquaredError, ExpectedCalibrationError
from tensorflow_probability import distributions as tfpd
from utils import pretty_model_name

# configure GPUs
for gpu in tf.config.list_physical_devices('GPU'):
    tf.config.experimental.set_memory_growth(gpu, enable=True)
tf.config.experimental.set_visible_devices(tf.config.list_physical_devices('GPU')[0], 'GPU')


def f_hidden_layers(d_in, d_hidden, **kwargs):
    assert isinstance(d_hidden, (list, tuple)) and all(isinstance(d, int) for d in d_hidden)
    nn = tf.keras.Sequential(layers=[tf.keras.layers.InputLayer(d_in)], name=kwargs.get('name'))
    for d in d_hidden:
        nn.add(tf.keras.layers.Dense(d, activation='elu'))
    return nn


def f_output_layer(d_in, d_out, f_out, **kwargs):
    assert isinstance(d_out, int) and d_out > 0
    return tf.keras.Sequential(name=kwargs.get('name'), layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Dense(units=d_out, activation=f_out),
    ])


def f_neural_net(d_in, d_out, d_hidden, f_out=None, **kwargs):
    m = f_hidden_layers(d_in, d_hidden, **kwargs)
    m.add(f_output_layer(d_in=d_hidden[-1], d_out=d_out, f_out=f_out, **kwargs))
    return m


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
        params = self.optimization_step(x, y)
        self.update_metrics(y, **params)

        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = self.parse_keras_inputs(data)

        # update metrics
        self.update_metrics(y, **self.call(x, training=True))

        return {m.name: m.result() for m in self.metrics}

    def update_metrics(self, y, **params):
        py_x = self.predictive_distribution(**params)
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), py_x.cdf(y))
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)


class Gaussian(Regression):

    def __init__(self, **kwargs):
        Regression.__init__(self, name=kwargs['name'])

    def predictive_distribution(self, *, x=None, mean=None, std=None):
        if mean is None or std is None:
            assert x is not None
            mean, std = self.call(x, training=False).values()
        return tfpd.Normal(loc=mean, scale=std)


class UnitVarianceGaussian(Gaussian, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        Regression.__init__(self, name='UnitVariance', **kwargs)
        self.likelihood = 'Normal'

        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.f_loc = f_param(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
        else:
            self.f_trunk = f_trunk(dim_x, **kwargs)
            dim_latent = self.f_trunk.output_shape[1:]
            self.f_loc = f_param(d_in=dim_latent, d_out=dim_y, f_out=None, name='f_mean', **kwargs)

    def call(self, x, **kwargs):
        mean = self.f_loc(self.f_trunk(x, **kwargs), **kwargs)
        return {'mean': mean, 'std': tf.ones_like(mean)}

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(loc=params['mean'], scale=1.0), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class HeteroscedasticGaussian(Gaussian, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        Regression.__init__(self, name=kwargs.pop('name', 'Heteroscedastic'), **kwargs)
        self.likelihood = 'Normal'

        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.f_loc = f_param(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
            self.f_scale = f_param(d_in=dim_x, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)
        else:
            self.f_trunk = f_trunk(dim_x, **kwargs)
            dim_latent = self.f_trunk.output_shape[1:]
            self.f_loc = f_param(d_in=dim_latent, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
            self.f_scale = f_param(d_in=dim_latent, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        return {'mean': self.f_loc(z, **kwargs), 'std': self.f_scale(z, **kwargs)}

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(*params.values()), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class SecondOrderMean(HeteroscedasticGaussian, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        HeteroscedasticGaussian.__init__(self, dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param,
                                         name='SecondOrderMean', **kwargs)
        self.likelihood = 'Normal'

    def optimization_step(self, x, y):

        # take necessary gradients
        with tf.GradientTape(persistent=True) as tape:
            params = self.call(x, training=True)
            params_concat = tf.concat(list(params.values()), axis=1)
            py_x = tfpd.Independent(tfpd.Normal(*params.values()), reinterpreted_batch_ndims=1)
            nll = -tf.reduce_mean(py_x.log_prob(y))
            regularization = tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        d_nll_d_params = tape.gradient(nll, params)

        # second order adjustment of gradient w.r.t. the mean
        d_nll_d_params['mean'] = d_nll_d_params['mean'] * params['std'] ** 2
        d_nll_d_params = tf.concat(list(d_nll_d_params.values()), axis=1)

        # finalize and apply gradients
        gradients = []
        for weight in self.trainable_variables:
            gradient = tape.gradient(params_concat, weight, d_nll_d_params)
            gradient += tape.gradient(regularization, weight, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            gradients.append(gradient)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return params


class FaithfulHeteroscedasticGaussian(HeteroscedasticGaussian, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        HeteroscedasticGaussian.__init__(self, dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param,
                                         name='FaithfulHeteroscedastic', **kwargs)
        self.likelihood = 'Normal'

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            z = self.f_trunk(x, training=True)
            mean = self.f_loc(z, training=True)
            std = self.f_scale(tf.stop_gradient(z), training=True)
            py_loc = tfpd.Independent(tfpd.Normal(loc=mean, scale=1.0), reinterpreted_batch_ndims=1)
            py_std = tfpd.Independent(tfpd.Normal(loc=tf.stop_gradient(mean), scale=std), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_loc.log_prob(y) + py_std.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'mean': mean, 'std': std}


class BetaNLL(HeteroscedasticGaussian, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, beta=0.5, **kwargs):
        HeteroscedasticGaussian.__init__(self, dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param,
                                         name='BetaNLL', **kwargs)
        self.likelihood = 'Normal'
        self.beta = beta

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Normal(*params.values())
            beta_nll = -py_x.log_prob(y) * tf.stop_gradient(params['std'] ** (2 * self.beta))
            loss = tf.reduce_sum(beta_nll) / tf.cast(tf.shape(beta_nll)[0], tf.float32)
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class HeteroscedasticStudent(Regression, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        Regression.__init__(self, name=kwargs.pop('name', 'HeteroscedasticStudent'), **kwargs)
        self.likelihood = 'Student'
        self.min_df = 3

        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.f_df = f_param(d_in=dim_x, d_out=dim_y, f_out='softplus', name='f_df', **kwargs)
            self.f_loc = f_param(d_in=dim_x, d_out=dim_y, f_out=None, name='f_loc', **kwargs)
            self.f_scale = f_param(d_in=dim_x, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)
        else:
            self.f_trunk = f_trunk(dim_x, **kwargs)
            dim_latent = self.f_trunk.output_shape[1:]
            self.f_df = f_param(d_in=dim_latent, d_out=dim_y, f_out='softplus', name='f_df', **kwargs)
            self.f_loc = f_param(d_in=dim_latent, d_out=dim_y, f_out=None, name='f_loc', **kwargs)
            self.f_scale = f_param(d_in=dim_latent, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        return {'df': self.min_df + self.f_df(z, **kwargs),
                'loc': self.f_loc(z, **kwargs),
                'scale': self.f_scale(z, **kwargs)}

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.StudentT(**params), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params

    def predictive_distribution(self, *, x=None, **params):
        if params.keys() != {'df', 'loc', 'scale'}:
            assert x is not None
            params = self.call(x, training=False)
        return tfpd.StudentT(**params)


class FaithfulHeteroscedasticStudent(HeteroscedasticStudent, ABC):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        HeteroscedasticStudent.__init__(self, dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param,
                                        name='FaithfulHeteroscedasticStudent', **kwargs)
        self.likelihood = 'Student'

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            z = self.f_trunk(x, training=True)
            df = self.min_df + self.f_df(tf.stop_gradient(z), training=True)
            loc = self.f_loc(z, training=True)
            scale = self.f_scale(tf.stop_gradient(z), training=True)
            py_loc = tfpd.Independent(tfpd.Normal(loc=loc, scale=1.0), 1)
            py_std = tfpd.Independent(tfpd.StudentT(df, tf.stop_gradient(loc), scale), 1)
            loss = -tf.reduce_mean(py_loc.log_prob(y) + py_std.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'df': df, 'loc': loc, 'scale': scale}


def get_models_and_configurations(nn_kwargs):
    return [
        dict(model=UnitVarianceGaussian, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=HeteroscedasticGaussian, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=SecondOrderMean, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=FaithfulHeteroscedasticGaussian, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=BetaNLL, model_kwargs=dict(beta=0.5), nn_kwargs=nn_kwargs),
        dict(model=BetaNLL, model_kwargs=dict(beta=1.0), nn_kwargs=nn_kwargs),
    ]


def fancy_plot(x_train, y_train, x_test, target_mean, target_std, predicted_mean, predicted_std, plot_title):
    # squeeze everything
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_test = np.squeeze(x_test)
    target_mean = np.squeeze(target_mean)
    target_std = np.squeeze(target_std)
    predicted_mean = np.squeeze(predicted_mean)
    predicted_std = np.squeeze(predicted_std)

    # clamp infinite values
    predicted_std[np.isinf(predicted_std)] = 1e6

    # get a new figure
    fig, ax = plt.subplots(nrows=2, figsize=(7.5, 5))
    fig.suptitle(plot_title)

    # plot the data
    sizes = 12.5 * np.ones_like(x_train)
    sizes[-2:] = 125
    ax[0].scatter(x_train, y_train, alpha=0.5, s=sizes)

    # plot the true mean and standard deviation
    ax[0].plot(x_test, target_mean, '--k')
    ax[0].plot(x_test, target_mean + target_std, ':k')
    ax[0].plot(x_test, target_mean - target_std, ':k')

    # plot the model's mean and standard deviation
    l = ax[0].plot(x_test, predicted_mean, label='predicted')[0]
    ax[0].fill_between(x_test[:, ], predicted_mean - predicted_std, predicted_mean + predicted_std,
                       color=l.get_color(), alpha=0.5)
    ax[0].plot(x_test, target_mean, '--k', label='truth')

    # clean it up
    ax[0].set_xlim([0, 10])
    ax[0].set_ylim([-12.5, 12.5])
    ax[0].set_ylabel('E[y|x]')
    ax[0].legend()

    # plot the std
    ax[1].plot(x_test, predicted_std, label='predicted')
    ax[1].plot(x_test, target_std, '--k', label='truth')
    ax[1].set_xlim([0, 10])
    ax[1].set_ylim([0, 6])
    ax[1].set_xlabel('x')
    ax[1].set_ylabel('std(y|x)')
    ax[1].legend()

    # save the figure
    os.makedirs('results', exist_ok=True)
    fig.savefig(os.path.join('results', 'toy_' + plot_title.replace(' ', '') + '.pdf'))


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--architecture', type=str, default='separate', help='network architecture')
    parser.add_argument('--beta', type=float, default=0.5, help='beta setting for BetaNLL')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--model', type=str, default='FaithfulHeteroscedastic', help='which model to use')
    parser.add_argument('--seed', type=int, default=12345, help='random number seed for reproducibility')
    args = parser.parse_args()

    # set random number seed
    tf.keras.utils.set_random_seed(args.seed)

    # generate data
    toy_data = generate_toy_data()

    # pick the appropriate model
    if args.model == 'UnitVarianceGaussian':
        args.architecture = 'single'
        model = UnitVarianceGaussian
    elif args.model == 'HeteroscedasticGaussian':
        model = HeteroscedasticGaussian
    elif args.model == 'SecondOrderMean':
        model = SecondOrderMean
    elif args.model == 'FaithfulHeteroscedasticGaussian':
        model = FaithfulHeteroscedasticGaussian
    elif args.model == 'BetaNLL':
        model = BetaNLL
    elif args.model == 'HeteroscedasticStudent':
        model = HeteroscedasticStudent
    elif args.model == 'FaithfulHeteroscedasticStudent':
        model = FaithfulHeteroscedasticStudent
    else:
        raise NotImplementedError
    assert args.architecture in {'single', 'separate', 'shared'}

    # declare model instance
    config = dict(d_hidden=(50,), beta=args.beta)
    if args.architecture in {'single', 'shared'}:
        model = model(dim_x=1, dim_y=1, f_trunk=f_hidden_layers, f_param=f_output_layer, **config)
    else:
        model = model(dim_x=1, dim_y=1, f_param=f_neural_net, **config)

    # build the model
    optimizer = tf.keras.optimizers.Adam(5e-3)
    model.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
        MeanLogLikelihood(),
        RootMeanSquaredError(),
        ExpectedCalibrationError(),
    ])

    # train model
    hist = model.fit(x=toy_data['x_train'], y=toy_data['y_train'],
                     batch_size=toy_data['x_train'].shape[0], epochs=int(20e3), verbose=0,
                     callbacks=[RegressionCallback(validation_freq=500, early_stop_patience=0)])

    # evaluate predictive model
    p_y_x = model.predictive_distribution(x=toy_data['x_test'])

    # plot results for toy data
    title = pretty_model_name(model, dict())
    fancy_plot(predicted_mean=p_y_x.mean().numpy(), predicted_std=p_y_x.stddev().numpy(), plot_title=title, **toy_data)
    plt.show()
