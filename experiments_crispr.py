import argparse
import os
import pickle

import models_regression as models
import numpy as np
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
    parser.add_argument('--seed', type=int, default=12345, help='number of trials per fold')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'crispr')
    os.makedirs(exp_path, exist_ok=True)

    # load and split data
    np.random.seed(args.seed)
    with open(os.path.join('data', 'junction', 'data.pkl'), 'rb') as f:
        x, y, nt_lut = pickle.load(f).values()
    x = tf.one_hot(x, depth=4)
    y = tf.expand_dims(y, axis=1)

    # loop over models
    metrics = pd.DataFrame()
    measurements = pd.DataFrame()
    for model in [models.UnitVarianceNormal, models.HeteroscedasticNormal, models.FaithfulHeteroscedasticNormal]:

        # set random seed
        tf.keras.utils.set_random_seed(args.seed)

        # initialize model
        model = model(dim_x=x.shape[1:], dim_y=1, f_param_net=f_param, f_trunk=f_trunk)
        model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                      run_eagerly=args.debug,
                      metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

        # fit model
        hist = model.fit(x=x, y=y, batch_size=args.batch_size, epochs=int(10e3), verbose=0,
                         callbacks=[RegressionCallback(early_stop_patience=0)])

        # index for this model
        model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
        index = pd.Index([model_name], name='Model')

    #         # measure performance
    #         py_x = model.predictive_distribution(x=data['x_test'])
    #         measurements = pd.concat([measurements, pd.DataFrame(
    #             data={'Epoch': epoch, 'x': data['x_test'][:, 0],
    #                   'Mean': py_x.mean()[:, 0], 'Std. Deviation': py_x.stddev()[:, 0]},
    #             index=index.repeat(data['x_test'].shape[0]))])
    #
    # # save performance measures
    # metrics.to_pickle(os.path.join(exp_path, 'metrics.pkl'))
    # measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
