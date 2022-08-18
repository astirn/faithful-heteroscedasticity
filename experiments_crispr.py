import argparse
import os
import pickle
import shap

import models_regression as models
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from metrics import RootMeanSquaredError, ExpectedCalibrationError


# sequence representation network
def f_trunk(dim_x):
    return tf.keras.Sequential(name='SequenceTrunk', layers=[
        tf.keras.layers.InputLayer(dim_x),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
        tf.keras.layers.Flatten()])


# parameter network
def f_param(d_in, **kwargs):
    return models.param_net(d_in=d_in, d_out=1, d_hidden=[128, 32])


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_folds', type=int, default=10, help='number of pre-validation folds')
    parser.add_argument('--seed', type=int, default=12345, help='number of trials per fold')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'crispr')
    os.makedirs(exp_path, exist_ok=True)

    # load data and sample fold assignments
    with open(os.path.join('data', 'junction', 'data.pkl'), 'rb') as f:
        x, y, nt_lut = pickle.load(f).values()
    x = tf.one_hot(x, depth=4)
    y = tf.expand_dims(y, axis=1)
    folds = tf.random.stateless_categorical(tf.ones([x.shape[0], args.num_folds]), num_samples=1, seed=2 * [args.seed])
    folds = tf.squeeze(folds)

    # loop over validation folds
    measurements = pd.DataFrame()
    for k in range(args.num_folds):
        print('******************** Fold {:d}/{:d} ********************'.format(k + 1, args.num_folds))

        # loop over models
        for mdl in [models.UnitVarianceNormal, models.HeteroscedasticNormal, models.FaithfulHeteroscedasticNormal]:

            # initialize model
            tf.keras.utils.set_random_seed(k * args.seed)
            model = mdl(dim_x=x.shape[1:], dim_y=1, f_param=f_param, f_trunk=f_trunk)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          run_eagerly=args.debug,
                          metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

            # fit model
            i_train, i_valid = tf.not_equal(folds, k), tf.equal(folds, k)
            hist = model.fit(x=x[i_train], y=y[i_train], validation_data=(x[i_valid], y[i_valid]),
                             batch_size=args.batch_size, epochs=int(10e3), verbose=0,
                             callbacks=[RegressionCallback(early_stop_patience=100)])

            # index for this model
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = pd.Index([model_name], name='Model')

            # save local performance measurements
            params = model.predict(x=x[i_valid], verbose=0)
            squared_errors = tf.reduce_sum((y[i_valid] - params['mean']) ** 2, axis=-1)
            cdf_y = tf.reduce_sum(model.predictive_distribution(**params).cdf(y[i_valid]), axis=-1)
            measurements = pd.concat([measurements, pd.DataFrame({
                'normalized': True,
                'squared errors': squared_errors,
                'F(y)': cdf_y,
            }, index.repeat(len(y[i_valid])))])

            # # compute SHAP values
            # num_background_samples = min(5000, x[i_train].shape[0])
            # background = tf.random.shuffle(x[i_train])[:num_background_samples]
            # e = shap.DeepExplainer(model, background)
            # shap_values = e.shap_values(x[i_valid].numpy())

    # save performance measures
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
