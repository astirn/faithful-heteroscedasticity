import argparse
import copy
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
    dropout_rate = kwargs.get('dropout_rate')
    nn = tf.keras.Sequential(layers=[tf.keras.layers.InputLayer(d_in)], name=kwargs.get('name'))
    for d in d_hidden:
        nn.add(tf.keras.layers.Dense(d, activation='elu'))
        if dropout_rate is not None:
            nn.add(tf.keras.layers.Dropout(rate=dropout_rate))
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
        super().__init__(name=kwargs['name'])

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
        self.update_metrics(y, **self.call(x, training=False))
        return {m.name: m.result() for m in self.metrics}

    def update_metrics(self, y, **params):
        py_x = self.predictive_distribution(**params)
        predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), py_x.cdf(y))
        self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)


class Normal(Regression):

    def __init__(self, **kwargs):
        super().__init__(name=kwargs['name'])

    @property
    def model_class(self):
        return 'Normal'

    def predictive_distribution(self, *, x=None, **params):
        if params.keys() != {'loc', 'scale'}:
            assert x is not None
            params = self.call(x, training=False)
        return tfpd.Normal(**params)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(**params), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class UnitVarianceNormal(Normal):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        super().__init__(name=kwargs.pop('name', 'UnitVarianceNormal'), **kwargs)
        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.dim_f_trunk = dim_x
        else:
            self.f_trunk = f_trunk(d_in=dim_x, name='f_trunk', **kwargs)
            self.dim_f_trunk = self.f_trunk.output_shape[1:]
        self.f_mean = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out=None, name='f_mean', **kwargs)

    def call(self, x, **kwargs):
        mean = self.f_mean(self.f_trunk(x, **kwargs), **kwargs)
        return {'loc': mean, 'scale': tf.ones_like(mean)}


class HeteroscedasticNormal(UnitVarianceNormal):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        name = kwargs.pop('name', 'HeteroscedasticNormal')
        super().__init__(dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param, name=name, **kwargs)
        self.f_scale = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        return {'loc': self.f_mean(z, **kwargs), 'scale': self.f_scale(z, **kwargs)}


class BetaNLL(HeteroscedasticNormal):

    def __init__(self, *, beta=0.5, **kwargs):
        super().__init__(name='BetaNLL', **kwargs)
        self.beta = beta

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Normal(*params.values())
            beta_nll = -py_x.log_prob(y) * tf.stop_gradient(params['scale'] ** (2 * self.beta))
            loss = tf.reduce_sum(beta_nll) / tf.cast(tf.shape(beta_nll)[0], tf.float32)
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class SecondOrderMean(HeteroscedasticNormal):

    def __init__(self, **kwargs):
        super().__init__(name='SecondOrderMean', **kwargs)

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
        d_nll_d_params['loc'] = d_nll_d_params['loc'] * params['scale'] ** 2
        d_nll_d_params = tf.concat(list(d_nll_d_params.values()), axis=1)

        # finalize and apply gradients
        gradients = []
        for weight in self.trainable_variables:
            gradient = tape.gradient(params_concat, weight, d_nll_d_params)
            gradient += tape.gradient(regularization, weight, unconnected_gradients=tf.UnconnectedGradients.ZERO)
            gradients.append(gradient)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        return params


class FaithfulNormal(Normal):

    def __int__(self, **kwargs):
        super().__init__(**kwargs)

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_loc = tfpd.Independent(tfpd.Normal(loc=params['loc'], scale=1.0), 1)
            py_std = tfpd.Independent(tfpd.Normal(loc=tf.stop_gradient(params['loc']), scale=params['scale']), 1)
            loss = -tf.reduce_mean(py_loc.log_prob(y) + py_std.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class FaithfulHeteroscedasticNormal(FaithfulNormal, HeteroscedasticNormal):

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop('name', 'FaithfulHeteroscedasticNormal'), **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        return {'loc': self.f_mean(z, **kwargs), 'scale': self.f_scale(tf.stop_gradient(z), **kwargs)}


class NormalMixture(object):

    def call(self, x, **kwargs):
        raise NotImplementedError

    def predict_step(self, x):
        params = self.call(x, training=False)
        params['loc'] = tf.transpose(params['loc'], [1, 0, 2])
        params['scale'] = tf.transpose(params['scale'], [1, 0, 2])
        params['batch_lead'] = tf.constant([True], dtype=tf.bool)
        return params

    def predictive_distribution(self, *, x=None, **params):
        if not {'loc', 'scale'}.issubset(set(params.keys())):
            assert x is not None
            params = self.call(x, training=False)
        if 'batch_lead' in params.keys():
            params.pop('batch_lead')
            params['loc'] = tf.transpose(params['loc'], [0, 2, 1])
            params['scale'] = tf.transpose(params['scale'], [0, 2, 1])
        else:
            params['loc'] = tf.transpose(params['loc'], [1, 2, 0])
            params['scale'] = tf.transpose(params['scale'], [1, 2, 0])
        return tfpd.MixtureSameFamily(
            mixture_distribution=tfpd.Categorical(logits=tf.ones(tf.shape(params['loc'])[-1])),
            components_distribution=tfpd.Normal(**params))


class MonteCarloDropout(NormalMixture, ABC):

    def __init__(self, *, dropout_rate=0.25, mc_samples=10):
        NormalMixture.__init__(self)
        self.dropout_rate = dropout_rate
        self.mc_samples = mc_samples

    @property
    def model_class(self):
        return 'Monte Carlo Dropout'

    def reshape_dims(self, x):
        return tf.stack([self.mc_samples, tf.shape(x)[0], -1])


class UnitVarianceMonteCarloDropout(MonteCarloDropout, UnitVarianceNormal):

    def __init__(self, **kwargs):
        MonteCarloDropout.__init__(self)
        kwargs.update(dict(dropout_rate=self.dropout_rate))
        UnitVarianceNormal.__init__(self, name='UnitVarianceMonteCarloDropout', **kwargs)

    def call(self, x, **kwargs):
        z = tf.concat([self.f_trunk(x, training=True) for _ in range(self.mc_samples)], axis=0)
        loc = tf.reshape(self.f_mean(z, training=True), self.reshape_dims(x))
        return {'loc': loc, 'scale': tf.ones(tf.shape(loc))}


class HeteroscedasticMonteCarloDropout(MonteCarloDropout, HeteroscedasticNormal):

    def __init__(self, **kwargs):
        MonteCarloDropout.__init__(self)
        kwargs.update(dict(dropout_rate=self.dropout_rate))
        HeteroscedasticNormal.__init__(self, name='HeteroscedasticMonteCarloDropout', **kwargs)

    def call(self, x, **kwargs):
        z = tf.concat([self.f_trunk(x, training=True) for _ in range(self.mc_samples)], axis=0)
        loc = tf.reshape(self.f_mean(z, training=True), self.reshape_dims(x))
        scale = tf.reshape(self.f_scale(z, training=True), self.reshape_dims(x))
        return {'loc': loc, 'scale': tf.clip_by_value(scale, 1e-9, np.inf)}


class FaithfulHeteroscedasticMonteCarloDropout(MonteCarloDropout, FaithfulHeteroscedasticNormal):

    def __init__(self, **kwargs):
        MonteCarloDropout.__init__(self)
        kwargs.update(dict(dropout_rate=self.dropout_rate))
        FaithfulHeteroscedasticNormal.__init__(self, name='FaithfulHeteroscedasticMonteCarloDropout', **kwargs)

    def call(self, x, **kwargs):
        z = tf.concat([self.f_trunk(x, training=True) for _ in range(self.mc_samples)], axis=0)
        loc = tf.reshape(self.f_mean(z, training=True), self.reshape_dims(x))
        scale = tf.reshape(self.f_scale(tf.stop_gradient(z), training=True), self.reshape_dims(x))
        return {'loc': loc, 'scale': tf.clip_by_value(scale, 1e-9, np.inf)}


class DeepEnsemble(NormalMixture, Normal, ABC):

    def __init__(self, *, num_models=10, **kwargs):
        NormalMixture.__init__(self)
        Normal.__init__(self, **kwargs)
        self.num_models = num_models

    @property
    def model_class(self):
        return 'Deep Ensemble'


class UnitVarianceDeepEnsemble(DeepEnsemble):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        super().__init__(name=kwargs.pop('name', 'UnitVarianceDeepEnsemble'), **kwargs)
        self.f_trunk = []
        self.f_loc = []
        for m in range(self.num_models):
            if f_trunk is None:
                self.f_trunk += [lambda x, **k: x]
                self.dim_f_trunk = dim_x
            else:
                self.f_trunk += [f_trunk(d_in=dim_x, name='f_trunk' + str(m), **kwargs)]
                self.dim_f_trunk = self.f_trunk[0].output_shape[1:]
            self.f_loc += [f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out=None, name='f_loc' + str(m), **kwargs)]

    def call(self, x, **kwargs):
        z = [f_trunk(x, **kwargs) for f_trunk in self.f_trunk]
        loc = tf.stack([f_loc(z[i], **kwargs) for i, f_loc in enumerate(self.f_loc)], axis=0)
        return {'loc': loc, 'scale': tf.ones(tf.shape(loc))}


class HeteroscedasticDeepEnsemble(UnitVarianceDeepEnsemble):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        name = kwargs.pop('name', 'HeteroscedasticDeepEnsemble')
        super().__init__(dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param, name=name, **kwargs)
        self.f_scale = []
        for m in range(self.num_models):
            self.f_scale += [f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)]

    def call(self, x, **kwargs):
        z = [f_trunk(x, **kwargs) for f_trunk in self.f_trunk]
        loc = tf.stack([f_loc(z[i], **kwargs) for i, f_loc in enumerate(self.f_loc)], axis=0)
        scale = tf.stack([f_scale(z[i], **kwargs) for i, f_scale in enumerate(self.f_scale)], axis=0)
        return {'loc': loc, 'scale': scale}


class FaithfulHeteroscedasticDeepEnsemble(FaithfulNormal, HeteroscedasticDeepEnsemble):

    def __init__(self, **kwargs):
        super().__init__(name=kwargs.pop('name', 'FaithfulHeteroscedasticDeepEnsemble'), **kwargs)

    def call(self, x, **kwargs):
        z = [f_trunk(x, **kwargs) for f_trunk in self.f_trunk]
        loc = tf.stack([f_loc(z[i], **kwargs) for i, f_loc in enumerate(self.f_loc)], axis=0)
        scale = tf.stack([f_scale(tf.stop_gradient(z[i]), **kwargs) for i, f_scale in enumerate(self.f_scale)], axis=0)
        return {'loc': loc, 'scale': scale}


class Student(Regression):

    def __init__(self, **kwargs):
        super().__init__(name=kwargs['name'])
        self.min_df = 3
        self.unit_variance_df = 100
        self.unit_variance_scale = ((self.unit_variance_df - 2) / self.unit_variance_df) ** 0.5

    @property
    def model_class(self):
        return 'Student'

    def predictive_distribution(self, *, x=None, **params):
        if params.keys() != {'df', 'loc', 'scale'}:
            assert x is not None
            params = self.call(x, training=False)
        return tfpd.StudentT(**params)


class UnitVarianceStudent(Student):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        super().__init__(name=kwargs.pop('name', 'UnitVarianceStudent'), **kwargs)
        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.dim_f_trunk = dim_x
        else:
            self.f_trunk = f_trunk(d_in=dim_x, name='f_trunk', **kwargs)
            self.dim_f_trunk = self.f_trunk.output_shape[1:]
        self.f_loc = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out=None, name='f_loc', **kwargs)

    def call(self, x, **kwargs):
        loc = self.f_loc(self.f_trunk(x, **kwargs), **kwargs)
        df = self.unit_variance_df * tf.ones(tf.shape(loc))
        scale = self.unit_variance_scale * tf.ones(tf.shape(loc))
        return {'df': df, 'loc': loc, 'scale': scale}

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.Normal(loc=params['loc'], scale=1.0), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class HeteroscedasticStudent(UnitVarianceStudent):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        name = kwargs.pop('name', 'HeteroscedasticStudent')
        super().__init__(dim_x=dim_x, dim_y=dim_y, f_trunk=f_trunk, f_param=f_param, name=name, **kwargs)
        self.f_df = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out='softplus', name='f_df', **kwargs)
        self.f_scale = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out='softplus', name='f_scale', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        df = self.min_df + self.f_df(z, **kwargs)
        loc = self.f_loc(z, **kwargs)
        scale = self.f_scale(z, **kwargs)
        return {'df': df, 'loc': loc, 'scale': tf.clip_by_value(scale, 1e-9, np.inf)}

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_x = tfpd.Independent(tfpd.StudentT(**params), reinterpreted_batch_ndims=1)
            loss = -tf.reduce_mean(py_x.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class FaithfulHeteroscedasticStudent(HeteroscedasticStudent):

    def __init__(self, **kwargs):
        HeteroscedasticStudent.__init__(self, name='FaithfulHeteroscedasticStudent', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        df = self.min_df + self.f_df(tf.stop_gradient(z), **kwargs)
        loc = self.f_loc(z, **kwargs)
        scale = self.f_scale(tf.stop_gradient(z), **kwargs)
        return {'df': df, 'loc': loc, 'scale': tf.clip_by_value(scale, 1e-9, np.inf)}

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:
            params = self.call(x, training=True)
            py_loc = tfpd.Independent(tfpd.Normal(loc=params['loc'], scale=1.0), 1)
            py_std = tfpd.Independent(tfpd.StudentT(params['df'], tf.stop_gradient(params['loc']), params['scale']), 1)
            loss = -tf.reduce_mean(py_loc.log_prob(y) + py_std.log_prob(y))
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
        return params


class VariationalVariance(Student):

    def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
        super().__init__(name=kwargs.pop('name'))
        self.k = 100
        self.num_mc_samples = 100

        # trunk network
        if f_trunk is None:
            self.f_trunk = lambda x, **k: x
            self.dim_f_trunk = dim_x
        else:
            self.f_trunk = f_trunk(d_in=dim_x, name='f_trunk', **kwargs)
            self.dim_f_trunk = self.f_trunk.output_shape[1:]

        # parameter head networks
        self.f_mu = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out=None, name='f_mu', **kwargs)
        self.f_alpha = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out='softplus', name='f_alpha', **kwargs)
        self.f_beta = f_param(d_in=self.dim_f_trunk, d_out=dim_y, f_out='softplus', name='f_beta', **kwargs)

    def call(self, x, **kwargs):
        z = self.f_trunk(x, **kwargs)
        mu = self.f_mu(z, **kwargs)
        alpha = self.min_df / 2 + self.f_alpha(z, **kwargs)
        beta = self.f_beta(z, **kwargs)
        return {'z': z, 'mu': mu, 'alpha': alpha, 'beta': beta}

    def predictive_distribution(self, *, x=None, **params):
        if not {'mu', 'alpha', 'beta'}.issubset(set(params.keys())):
            assert x is not None
            params = self.call(x, training=False)
        return tfpd.StudentT(df=2 * params['alpha'], loc=params['mu'], scale=tf.sqrt(params['beta'] / params['alpha']))

    def kl_divergence(self, z, alpha, beta, **kwargs):
        raise NotImplementedError

    def optimization_step(self, x, y):
        with tf.GradientTape() as tape:

            # run parameter networks
            params = self.call(x, training=True)

            # expected log likelihood
            precision = params['alpha'] / params['beta']
            log_precision = tf.math.digamma(params['alpha']) - tf.math.log(params['beta'])
            ell = 0.5 * (log_precision - tf.math.log(2 * np.pi) - (y - params['mu']) ** 2 * precision)
            ell = tf.reduce_sum(ell, axis=-1)

            # KL divergence
            dkl = self.kl_divergence(params['z'], params['alpha'], params['beta'], training=False)

            # negative ELBO loss
            loss = -tf.reduce_mean(ell - dkl)

            # layer losses
            loss += tf.reduce_sum(tf.stack(self.losses)) / tf.cast(tf.shape(x)[0], tf.float32)

        # apply gradients
        self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))

        return params


class VBEM(VariationalVariance):

    def __init__(self, **kwargs):
        super().__init__(name='VBEM*', **kwargs)
        u = tf.random.uniform(shape=(self.k, kwargs['dim_y']), minval=-3, maxval=3, dtype=tf.float32)
        v = tf.random.uniform(shape=(self.k, kwargs['dim_y']), minval=-3, maxval=3, dtype=tf.float32)
        self.u = tf.Variable(initial_value=u, dtype=tf.float32, trainable=True, name='u')
        self.v = tf.Variable(initial_value=v, dtype=tf.float32, trainable=True, name='v')
        self.f_pi = kwargs['f_param'](d_in=self.dim_f_trunk, d_out=self.k, f_out='softmax', name='f_pi', **kwargs)

    def kl_divergence(self, z, alpha, beta, **kwargs):

        # variational family
        qp = tfpd.Independent(tfpd.Gamma(alpha, beta), reinterpreted_batch_ndims=1)
        p_samples = tf.expand_dims(qp.sample(sample_shape=self.num_mc_samples), axis=2)
        p_samples = tf.clip_by_value(p_samples, clip_value_min=1e-30, clip_value_max=tf.float32.max)

        # log prior probabilities for each component--shape is [# MC samples, batch size, # components]
        a = tf.reshape(tf.nn.softplus(self.u), tf.stack([1, 1, self.k, -1]))
        b = tf.reshape(tf.nn.softplus(self.v), tf.stack([1, 1, self.k, -1]))
        log_pp_c = tfpd.Independent(tfpd.Gamma(a, b), 1).log_prob(p_samples)

        # prior mixture proportions--shape is [batch size, # components]
        pi = self.f_pi(z)

        # take the expectation w.r.t. to mixture proportions--shape will be [# MC samples, batch size]
        epsilon = 1e-30
        log_pp = tf.reduce_logsumexp(tf.math.log(pi + epsilon) + log_pp_c, axis=-1)  # add offset to avoid log(0)
        log_pp -= tf.math.log(tf.reduce_sum(epsilon + pi, axis=-1))  # correct for the offset

        # average over MC samples--shape will be [batch size]
        dkl = -qp.entropy() - tf.reduce_mean(log_pp, axis=0)

        return dkl


def get_models_and_configurations(nn_kwargs, mcd_kwargs=None):

    # Normal models
    models = [
        dict(model=UnitVarianceNormal, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=HeteroscedasticNormal, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=FaithfulHeteroscedasticNormal, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=SecondOrderMean, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=BetaNLL, model_kwargs=dict(beta=0.5), nn_kwargs=nn_kwargs),
        dict(model=BetaNLL, model_kwargs=dict(beta=1.0), nn_kwargs=nn_kwargs),
    ]

    # Monte Carlo Dropout models
    if mcd_kwargs is not None:
        mcd_nn_kwargs = copy.deepcopy(nn_kwargs)  # these need to layers to work reasonably
        if 'd_hidden' in mcd_nn_kwargs.keys() and len(mcd_nn_kwargs.get('d_hidden')) == 1:
            mcd_nn_kwargs.update(dict(d_hidden=2 * mcd_nn_kwargs['d_hidden']))
        models += [
            dict(model=UnitVarianceMonteCarloDropout, model_kwargs=mcd_kwargs, nn_kwargs=mcd_nn_kwargs),
            dict(model=HeteroscedasticMonteCarloDropout, model_kwargs=mcd_kwargs, nn_kwargs=mcd_nn_kwargs),
            dict(model=FaithfulHeteroscedasticMonteCarloDropout, model_kwargs=mcd_kwargs, nn_kwargs=mcd_nn_kwargs),
        ]

    # Deep Ensemble models
    models += [
        dict(model=UnitVarianceDeepEnsemble, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=HeteroscedasticDeepEnsemble, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=FaithfulHeteroscedasticDeepEnsemble, model_kwargs=dict(), nn_kwargs=nn_kwargs),
    ]

    # Student models
    models += [
        dict(model=UnitVarianceStudent, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=HeteroscedasticStudent, model_kwargs=dict(), nn_kwargs=nn_kwargs),
        dict(model=FaithfulHeteroscedasticStudent, model_kwargs=dict(), nn_kwargs=nn_kwargs),
    ]

    return models


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
    parser.add_argument('--model', type=str, default='FaithfulHeteroscedasticNormal', help='which model to use')
    parser.add_argument('--seed', type=int, default=12345, help='random number seed for reproducibility')
    args = parser.parse_args()

    # set random number seed
    tf.keras.utils.set_random_seed(args.seed)

    # generate data
    toy_data = generate_toy_data()

    # pick the appropriate model
    if args.model == 'UnitVarianceNormal':
        args.architecture = 'single'
        model = UnitVarianceNormal
    elif args.model == 'HeteroscedasticNormal':
        model = HeteroscedasticNormal
    elif args.model == 'FaithfulHeteroscedasticNormal':
        model = FaithfulHeteroscedasticNormal
    elif args.model == 'SecondOrderMean':
        model = SecondOrderMean
    elif args.model == 'BetaNLL':
        model = BetaNLL
    elif args.model == 'UnitVarianceMonteCarloDropout':
        model = UnitVarianceMonteCarloDropout
    elif args.model == 'HeteroscedasticMonteCarloDropout':
        model = HeteroscedasticMonteCarloDropout
    elif args.model == 'FaithfulHeteroscedasticMonteCarloDropout':
        model = FaithfulHeteroscedasticMonteCarloDropout
    elif args.model == 'UnitVarianceDeepEnsemble':
        model = UnitVarianceDeepEnsemble
    elif args.model == 'HeteroscedasticDeepEnsemble':
        model = HeteroscedasticDeepEnsemble
    elif args.model == 'FaithfulHeteroscedasticDeepEnsemble':
        model = FaithfulHeteroscedasticDeepEnsemble
    elif args.model == 'UnitVarianceStudent':
        model = UnitVarianceStudent
    elif args.model == 'HeteroscedasticStudent':
        model = HeteroscedasticStudent
    elif args.model == 'FaithfulHeteroscedasticStudent':
        model = FaithfulHeteroscedasticStudent
    elif args.model == 'VBEM':
        model = VBEM
    else:
        raise NotImplementedError
    assert args.architecture in {'single', 'separate', 'shared'}

    # declare model instance
    config = dict(d_hidden=(50,) * (2 if 'MonteCarloDropout' in args.model else 1), beta=args.beta)
    if args.architecture in {'single', 'shared'}:
        model = model(dim_x=1, dim_y=1, f_trunk=f_hidden_layers, f_param=f_output_layer, **config)
    else:
        model = model(dim_x=1, dim_y=1, f_param=f_neural_net, **config)

    # build the model
    optimizer = tf.keras.optimizers.Adam(1e-3)
    model.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
        MeanLogLikelihood(),
        RootMeanSquaredError(),
        ExpectedCalibrationError(),
    ])

    # train model
    hist = model.fit(x=toy_data['x_train'], y=toy_data['y_train'],
                     batch_size=toy_data['x_train'].shape[0], epochs=int(30e3), verbose=0,
                     callbacks=[RegressionCallback(validation_freq=500, early_stop_patience=0)])

    # evaluate predictive model
    p_y_x = model.predictive_distribution(x=toy_data['x_test'])

    # plot results for toy data
    title = pretty_model_name(model, dict())
    fancy_plot(predicted_mean=p_y_x.mean().numpy(), predicted_std=p_y_x.stddev().numpy(), plot_title=title, **toy_data)
    plt.show()
