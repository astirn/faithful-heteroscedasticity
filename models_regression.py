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


def param_net(*, d_in, d_out, d_hidden=(50, ), f_hidden='elu', rate=0.0, f_out=None, name=None, **kwargs):
    assert isinstance(d_in, int) and d_in > 0
    assert isinstance(d_hidden, (list, tuple)) and all(isinstance(d, int) for d in d_hidden)
    assert isinstance(d_out, int) and d_out > 0
    nn = tf.keras.Sequential(name=name)
    nn.add(tf.keras.layers.InputLayer(d_in))
    for d in d_hidden:
        nn.add(tf.keras.layers.Dense(d, f_hidden))
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


class UnitVarianceNormal(Regression, ABC):

    def __init__(self, dim_x, dim_y, f_param, f_trunk=None, **kwargs):
        Regression.__init__(self, name='UnitVarianceNormal', **kwargs)

        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.f_mean = f_param(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
        else:
            self.f_trunk = f_trunk(dim_x)
            dim_latent = self.f_trunk.output_shape[1:]
            assert len(dim_latent) == 1
            self.f_mean = f_param(d_in=dim_latent[0], d_out=dim_y, f_out=None, name='f_mean', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        return {'mean': self.f_mean(z, **kwargs)}

    def predictive_distribution(self, *, x=None, mean=None):
        if mean is None:
            assert x is not None
            mean, = self.call(x, training=False).values()
        return tfpd.Normal(loc=mean, scale=1.0)

    def update_metrics(self, y, mean):
        py_x = self.predictive_distribution(mean=mean)
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), py_x.cdf(y))
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(loc=params['mean'], scale=1.0), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class HeteroscedasticNormal(Regression, ABC):

    def __init__(self, dim_x, dim_y, f_param, f_trunk=None, **kwargs):
        Regression.__init__(self, name=kwargs.pop('name', 'HeteroscedasticNormal'), **kwargs)

        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.f_mean = f_param(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
            self.f_scale = f_param(d_in=dim_x, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)
        else:
            self.f_trunk = f_trunk(dim_x)
            dim_latent = self.f_trunk.output_shape[1:]
            assert len(dim_latent) == 1
            self.f_mean = f_param(d_in=dim_latent[0], d_out=dim_y, f_out=None, name='f_mean', **kwargs)
            self.f_scale = f_param(d_in=dim_latent[0], d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        return {'mean': self.f_mean(z, **kwargs), 'std': self.f_scale(z, **kwargs)}

    def predictive_distribution(self, *, x=None, mean=None, std=None):
        if mean is None or std is None:
            assert x is not None
            mean, std = self.call(x, training=False).values()
        return tfpd.Normal(loc=mean, scale=std)

    def update_metrics(self, y, mean, std):
        py_x = self.predictive_distribution(mean=mean, std=std)
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), py_x.cdf(y))
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(*params.values()), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class FaithfulHeteroscedasticNormal(HeteroscedasticNormal, ABC):

    def __init__(self, dim_x, dim_y, f_param, f_trunk=None, **kwargs):
        HeteroscedasticNormal.__init__(self, dim_x, dim_y, f_param, f_trunk, name='FaithfulHeteroscedasticNormal', **kwargs)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            z = self.f_trunk(x, training=True)
            mean = self.f_mean(z, training=True)
            std = self.f_scale(tf.stop_gradient(z), training=True)
            py_loc = tfpd.Independent(tfpd.Normal(loc=mean, scale=1.0), reinterpreted_batch_ndims=1)
            py_std = tfpd.Independent(tfpd.Normal(loc=tf.stop_gradient(mean), scale=std), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_loc.log_prob(y) + py_std.log_prob(y))
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return {'mean': mean, 'std': std}


def fancy_plot(x_train, y_train, x_test, target_mean, target_std, predicted_mean, predicted_std, title):
    # squeeze everything
    x_train = np.squeeze(x_train)
    y_train = np.squeeze(y_train)
    x_test = np.squeeze(x_test)
    target_mean = np.squeeze(target_mean)
    target_std = np.squeeze(target_std)
    predicted_mean = np.squeeze(predicted_mean)
    predicted_std = np.squeeze(predicted_std)

    # get a new figure
    fig, ax = plt.subplots(2, 1)
    fig.suptitle(title)

    # plot the data
    sns.scatterplot(x_train, y_train, ax=ax[0])

    # plot the true mean and standard deviation
    ax[0].plot(x_test, target_mean, '--k')
    ax[0].plot(x_test, target_mean + target_std, ':k')
    ax[0].plot(x_test, target_mean - target_std, ':k')

    # plot the model's mean and standard deviation
    l = ax[0].plot(x_test, predicted_mean)[0]
    ax[0].fill_between(x_test[:, ], predicted_mean - predicted_std, predicted_mean + predicted_std, color=l.get_color(), alpha=0.5)
    ax[0].plot(x_test, target_mean, '--k')

    # clean it up
    ax[0].set_xlim([0, 10])
    ax[0].set_ylim([-12.5, 12.5])
    ax[0].set_ylabel('y')

    # plot the std
    ax[1].plot(x_test, predicted_std, label='predicted')
    ax[1].plot(x_test, target_std, '--k', label='truth')
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
    parser.add_argument('--model', type=str, default='FaithfulHeteroscedasticNormal', help='which model to use')
    args = parser.parse_args()

    # load data
    toy_data = generate_toy_data(num_samples=500)

    # pick the appropriate model
    plot_title = args.model
    if args.model == 'UnitVarianceNormal':
        MODEL = UnitVarianceNormal
    elif args.model == 'HeteroscedasticNormal':
        MODEL = HeteroscedasticNormal
    elif args.model == 'FaithfulHeteroscedasticNormal':
        MODEL = FaithfulHeteroscedasticNormal
    else:
        raise NotImplementedError

    # declare model instance
    mdl = MODEL(dim_x=toy_data['x_train'].shape[1], dim_y=toy_data['y_train'].shape[1], f_param=param_net)

    # build the model
    optimizer = tf.keras.optimizers.Adam(5e-3)
    mdl.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
        MeanLogLikelihood(),
        RootMeanSquaredError(),
        ExpectedCalibrationError(),
    ])

    # train model
    hist = mdl.fit(x=toy_data['x_train'], y=toy_data['y_train'],
                   batch_size=toy_data['x_train'].shape[0], epochs=int(20e3), verbose=0,
                   callbacks=[RegressionCallback(validation_freq=500, early_stop_patience=0)])

    # evaluate predictive model
    mdl.num_mc_samples = 2000
    p_y_x = mdl.predictive_distribution(x=toy_data['x_test'])

    # plot results for toy data
    fancy_plot(predicted_mean=p_y_x.mean().numpy(), predicted_std=p_y_x.stddev().numpy(), title=args.model, **toy_data)
    plt.show()
