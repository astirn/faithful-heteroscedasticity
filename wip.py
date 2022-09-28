# class VariationalEmpiricalBayes(Regression):
#
#     def __init__(self, *, dim_x, dim_y, f_trunk=None, f_param, **kwargs):
#         name = 'VariationalEmpiricalBayes'
#         Regression.__init__(self, name=name, **kwargs)
#
#         # precision prior
#         self.a = tf.Variable([2.0] * dim_y, trainable=True)
#         self.b = tf.Variable([1.0] * dim_y, trainable=True)
#
#         if f_trunk is None:
#             self.f_trunk = lambda x, **k: x
#             self.f_mean = f_param(d_in=dim_x, d_out=dim_y, f_out=None, name='f_mean', **kwargs)
#             self.f_alpha = f_param(d_in=dim_x, d_out=dim_y, f_out=self.f_out_alpha, name='f_alpha', **kwargs)
#             self.f_beta = f_param(d_in=dim_x, d_out=dim_y, f_out='softplus', name='f_beta', **kwargs)
#         else:
#             self.f_trunk = f_trunk(dim_x, **kwargs)
#             dim_latent = self.f_trunk.output_shape[1:]
#             assert len(dim_latent) == 1
#             self.f_mean = f_param(d_in=dim_latent[0], d_out=dim_y, f_out=None, name='f_mean', **kwargs)
#             self.f_alpha = f_param(d_in=dim_latent[0], d_out=dim_y, f_out=self.f_out_alpha, name='f_alpha', **kwargs)
#             self.f_beta = f_param(d_in=dim_latent[0], d_out=dim_y, f_out='softplus', name='f_beta', **kwargs)
#
#     @staticmethod
#     def f_out_alpha(x):
#         return 1 + tf.nn.softplus(x)
#
#     def call(self, x, **kwargs):
#         z = self.f_trunk(x, **kwargs)
#         return {'mean': self.f_mean(z, **kwargs), 'alpha': self.f_alpha(z, **kwargs), 'beta': self.f_beta(z, **kwargs)}
#
#     def optimization_step(self, x, y):
#
#         with tf.GradientTape(persistent=True) as tape:
#
#             # amortized parameter networks
#             params = self.call(x, training=True)
#
#             # squared errors
#             squared_errors = (y - params['mean']) ** 2
#
#             # empirical bayes prior and its log likelihood
#             a, b = 1 + tf.nn.softplus(self.a), tf.nn.softplus(self.b)
#             p_lambda = tfpd.Independent(tfpd.Gamma(tf.stop_gradient(a), tf.stop_gradient(b)), 1)
#             pll = tfpd.Independent(tfpd.InverseGamma(a, b), 1).log_prob(tf.stop_gradient(squared_errors))
#
#             # variational family
#             qp = tfpd.Independent(tfpd.Gamma(params['alpha'], params['beta']), reinterpreted_batch_ndims=1)
#
#             # use negative evidence lower bound as minimization objective
#             expected_lambda = params['alpha'] / params['beta']
#             expected_ln_lambda = tf.math.digamma(params['alpha']) - tf.math.log(params['beta'])
#             ell = 0.5 * (expected_ln_lambda - tf.math.log(2 * np.pi) - squared_errors * expected_lambda)
#             dkl = qp.kl_divergence(p_lambda)
#             loss = -tf.reduce_mean(tf.reduce_sum(ell, axis=-1) - dkl + pll)
#
#         # update model parameters
#         self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
#
#         return params
#
#     def predictive_distribution(self, *, x=None, mean=None, alpha=None, beta=None):
#         if mean is None or alpha is None or beta is None:
#             assert x is not None
#             mean, alpha, beta = self.call(x, training=False).values()
#         return tfpd.StudentT(df=2 * alpha, loc=mean, scale=tf.sqrt(beta / alpha))


# elif self.optimization in {'second-order-diag', 'second-order-full'}:
#     diag = 'diag' in self.optimization
#     gradients, network_params = self.second_order_gradients_diag(x, y, self.f_mean, self.f_precision, diag)
#     self.optimizer.apply_gradients(zip(gradients, network_params))


# def second_order_gradients_diag(self, x, y, f_mean, f_precision, diag):
#     # take necessary gradients
#     dim_batch = tf.cast(tf.shape(x)[0], tf.float32)
#     trainable_variables = f_mean.trainable_variables + f_precision.trainable_variables
#     with tf.GradientTape(persistent=True) as tape2:
#         with tf.GradientTape(persistent=True) as tape1:
#             mean, precision = f_mean.call(x, training=True), f_precision.call(x, training=True)
#             mean_precision = tf.stack([mean, precision], axis=-1)
#             py_x = tfpd.MultivariateNormalDiag(loc=mean, scale_diag=precision ** -0.5)
#             loss = tf.reduce_mean(-py_x.log_prob(self.whiten_targets(y)), axis=-1)
#         dl_dm = tape1.gradient(loss, mean)
#         dl_dp = tape1.gradient(loss, precision)
#         dmp_dnet = tape1.jacobian(mean_precision, trainable_variables)
#     d2nll_dm2 = tape2.gradient(dl_dm, mean) * dim_batch
#     d2nll_dp2 = tape2.gradient(dl_dp, precision) * dim_batch
#     # tf.assert_greater(d2nll_dp2, 0.0)
#     d2nll_dp2 = tf.clip_by_value(d2nll_dp2, 1e-3, np.inf)
#
#     # apply second order information
#     if diag:
#         dl_dmv = tf.stack([dl_dm / d2nll_dm2, dl_dp / d2nll_dp2], axis=-1)
#     else:
#         d2nll_dmdp = tape2.gradient(dl_dm, precision) * dim_batch
#         dim_H = tf.stack([-1, 2 * tf.shape(y)[-1], 2 * tf.shape(y)[-1]])
#         H = tf.reshape(tf.concat([10 * d2nll_dm2, d2nll_dmdp, d2nll_dmdp, 10 * d2nll_dp2], axis=-1), dim_H)
#         dl_dmv = tf.transpose(tf.linalg.solve(H, tf.stack([dl_dm, dl_dp], axis=-2)), [0, 2, 1])
#     gradients = [tf.tensordot(dl_dmv, d, axes=[[0, 1, 2], [0, 1, 2]]) for d in dmp_dnet]
#
#     return gradients, trainable_variables


# class MonteCarloDropout(HeteroscedasticRegression, ABC):
#
#     def __init__(self, dim_x, dim_y, num_mc_samples, **kwargs):
#         HeteroscedasticRegression.__init__(self, name='MonteCarloDropout', **kwargs)
#
#         # save configuration
#         self.num_mc_samples = num_mc_samples
#
#         # define parameter networks
#         self.f_mean = param_net(d_in=dim_x, d_out=dim_y, f_out=None, rate=0.1, name='mu', **kwargs)
#         self.f_precision = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', rate=0.1, name='lambda', **kwargs)
#
#     def call(self, inputs, **kwargs):
#         return self.f_mean(inputs['x'], **kwargs), self.f_precision(inputs['x'], **kwargs)
#
#     def predictive_central_moments(self, x):
#         means = tf.stack([self.f_mean(x, training=True) for _ in range(self.num_mc_samples)], axis=0)
#         variances = tf.stack([self.f_precision(x, training=True) ** -1 for _ in range(self.num_mc_samples)], axis=0)
#         predictive_mean = tf.reduce_mean(means, axis=0)
#         predictive_variance = tf.reduce_mean(means ** 2 + variances, axis=0) - tf.reduce_mean(means, axis=0) ** 2
#
#         return self.de_whiten_mean(predictive_mean), self.de_whiten_variance(predictive_variance)
#
#     def predictive_distribution(self, x):
#         raise NotImplementedError


# class DeepEnsemble(HeteroscedasticRegression, ABC):
#
#     def __init__(self, dim_x, dim_y, num_ensembles, **kwargs):
#         HeteroscedasticRegression.__init__(self, name='DeepEnsemble', **kwargs)
#
#         # define parameter networks
#         self.f_mean, self.f_precision = [], []
#         for i in range(num_ensembles):
#             s = str(i + 1)
#             self.f_mean += [param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu_' + s, **kwargs)]
#             self.f_precision += [param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='lambda_' + s, **kwargs)]
#
#     def call(self, inputs, **kwargs):
#         means = tf.stack([mean(inputs['x'], **kwargs) for mean in self.f_mean], axis=0)
#         precisions = tf.stack([precision(inputs['x'], **kwargs) for precision in self.f_precision], axis=0)
#         return means, precisions
#
#     def predictive_central_moments(self, x):
#         means = tf.stack([mean(x, training=False) for mean in self.f_mean], axis=0)
#         variances = tf.stack([precision(x, training=False) ** -1 for precision in self.f_precision], axis=0)
#         predictive_mean = tf.reduce_mean(means, axis=0)
#         predictive_variance = tf.reduce_mean(means ** 2 + variances, axis=0) - tf.reduce_mean(means, axis=0) ** 2
#
#         return self.de_whiten_mean(predictive_mean), self.de_whiten_variance(predictive_variance)
#
#     def predictive_distribution(self, x):
#         raise NotImplementedError


# class Student(Regression):
#
#     def __init__(self, dim_x, dim_y, **kwargs):
#         Regression.__init__(self, name='Student', **kwargs)
#
#         # parameter networks
#         self.mu = param_net(d_in=dim_x, d_out=dim_y, f_out=None, name='mu', **kwargs)
#         self.alpha = param_net(d_in=dim_x, d_out=dim_y, f_out=lambda x: 1 + tf.nn.softplus(x), name='alpha', **kwargs)
#         self.beta = param_net(d_in=dim_x, d_out=dim_y, f_out='softplus', name='beta', **kwargs)
#
#     def call(self, x, **kwargs):
#         return self.mu(x, **kwargs), self.alpha(x, **kwargs), self.beta(x, **kwargs)
#
#     def predictive_distribution(self, *args):
#         mu, alpha, beta = self.call(args[0], training=False) if len(args) == 1 else args
#         loc = self.de_whiten_mean(mu)
#         scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
#         return tfpd.StudentT(df=2 * alpha, loc=loc, scale=scale)
#
#     def update_metrics(self, y, mu, alpha, beta):
#         py_x = self.predictive_distribution(mu, alpha, beta)
#         scale = self.de_whiten_stddev(tf.sqrt(beta / alpha))
#         prob_errors = tfpd.StudentT(df=2 * alpha, loc=0, scale=1).cdf((y - py_x.mean()) / scale)
#         predictor_values = pack_predictor_values(py_x.mean(), py_x.log_prob(y), prob_errors)
#         self.compiled_metrics.update_state(y_true=y, y_pred=predictor_values)
#
#     def optimization_step(self, x, y):
#
#         with tf.GradientTape() as tape:
#
#             # amortized parameter networks
#             mu, alpha, beta = self.call(x, training=True)
#
#             # minimize negative log likelihood
#             py_x = tfpd.StudentT(df=2 * alpha, loc=mu, scale=tf.sqrt(beta / alpha))
#             ll = tf.reduce_sum(py_x.log_prob(self.whiten_targets(y)), axis=-1)
#             loss = tf.reduce_mean(-ll)
#
#         # update model parameters
#         self.optimizer.apply_gradients(zip(tape.gradient(loss, self.trainable_variables), self.trainable_variables))
#
#         return mu, alpha, beta


# def crispr_convergence_plots():
#     if not os.path.exists(os.path.join('experiments', 'crispr')):
#         return
#
#     # loop over datasets with learning metrics
#     for dataset in os.listdir(os.path.join('experiments', 'crispr')):
#         metrics_file = os.path.join('experiments', 'crispr', dataset, 'metrics.pkl')
#         if os.path.exists(metrics_file):
#             df_metrics = pd.read_pickle(metrics_file).reset_index().set_index('Observation')
#
#             # plot learning curves
#             fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
#             for i, index in enumerate(df_metrics.index.unique()):
#                 df = df_metrics.loc[index]
#                 sns.lineplot(data=df, x='Epoch', y='RMSE', hue='Fold', style='Model', ax=ax[0, i])
#                 sns.lineplot(data=df, x='Epoch', y='ECE', hue='Fold', style='Model', ax=ax[1, i])
#             plt.tight_layout()
#             # fig_learning_curve.savefig(os.path.join('results', 'toy_learning_curve.pdf'))
#
