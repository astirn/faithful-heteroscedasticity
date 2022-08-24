import argparse
import os
import pickle

import models_regression as models
import numpy as np
import pandas as pd
import tensorflow as tf

from data import generate_toy_data
from metrics import RootMeanSquaredError, ExpectedCalibrationError


# parameter network
def f_param(d_in, d_out, **kwargs):
    return models.param_net(d_in=d_in, d_out=d_out, d_hidden=[50], **kwargs)


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_seed', type=int, default=112358, help='seed to generate data')
    parser.add_argument('--epoch_modulo', type=int, default=2000, help='number of epochs between logging results')
    parser.add_argument('--epochs', type=int, default=30000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'convergence')
    os.makedirs(exp_path, exist_ok=True)

    # generate data
    data = generate_toy_data(seed=args.data_seed)
    with open(os.path.join(exp_path, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    # initialize mean model
    mean_model = models.UnitVarianceNormal(data['x_train'].shape[1], data['y_train'].shape[1], f_param)
    mean_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                       metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

    # initialize heteroscedastic model such that it starts with the same mean network initialization
    full_model = models.HeteroscedasticNormal(data['x_train'].shape[1], data['y_train'].shape[1], f_param)
    full_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                       metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])
    full_model.f_mean.set_weights(mean_model.get_weights())

    # initialize faithful heteroscedastic model such that it starts with the same mean/std network initializations
    faith_model = models.FaithfulHeteroscedasticNormal(data['x_train'].shape[1], data['y_train'].shape[1], f_param)
    faith_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                        metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])
    faith_model.f_mean.set_weights(mean_model.get_weights())
    faith_model.f_scale.set_weights(full_model.f_scale.get_weights())

    # loop over the epochs
    metrics = pd.DataFrame()
    measurements = pd.DataFrame()
    for epoch in range(0, args.epochs + 1, args.epoch_modulo):

        # loop over the models
        for model in [mean_model, full_model, faith_model]:
            print('\rEpoch {:d} of {:d}: {:s}'.format(epoch, args.epochs, model.name), end='')

            # index for this model
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = pd.Index([model_name], name='Model')

            # improve fit by specified number of epochs
            if epoch > 0:
                hist = model.fit(x=data['x_train'], y=data['y_train'],
                                 batch_size=data['x_train'].shape[0], epochs=args.epoch_modulo, verbose=0)
                metrics = pd.concat([metrics, pd.DataFrame(
                    data={'Epoch': epoch + np.array(hist.epoch),
                          'RMSE': hist.history['RMSE'],
                          'ECE': hist.history['ECE']},
                    index=index.repeat(args.epoch_modulo))])

            # measure performance
            py_x = model.predictive_distribution(x=data['x_test'])
            measurements = pd.concat([measurements, pd.DataFrame(
                data={'Epoch': epoch, 'x': data['x_test'][:, 0],
                      'Mean': py_x.mean()[:, 0], 'Std. Deviation': py_x.stddev()[:, 0]},
                index=index.repeat(data['x_test'].shape[0]))])

    # save performance measures
    metrics.to_pickle(os.path.join(exp_path, 'metrics.pkl'))
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
