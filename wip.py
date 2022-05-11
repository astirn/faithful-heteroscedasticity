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


# elif args.model == 'MonteCarloDropout':
#     MODEL = MonteCarloDropout
# elif args.model == 'DeepEnsemble':
#     MODEL = DeepEnsemble
