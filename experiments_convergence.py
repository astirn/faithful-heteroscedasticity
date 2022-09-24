import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import generate_toy_data
from metrics import RootMeanSquaredError, ExpectedCalibrationError
from models import UnitVariance, Heteroscedastic, SecondOrderMean, FaithfulHeteroscedastic
from utils import pretty_model_name


# hidden layer
def f_hidden_layer(d_in, d_hidden, **kwargs):
    return tf.keras.Sequential(name=kwargs.get('name'), layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Dense(units=d_hidden, activation='elu'),
    ])


# output layer
def f_output_layer(d_in, d_out, f_out, **kwargs):
    return tf.keras.Sequential(name=kwargs.get('name'), layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Dense(units=d_out, activation=f_out),
    ])


# single layer neural network
def f_neural_net(d_in, d_out, d_hidden, f_out, **kwargs):
    m = f_hidden_layer(d_in, d_hidden, **kwargs)
    m.add(f_output_layer(d_in=m.output_shape[1], d_out=d_out, f_out=f_out, **kwargs))
    return m


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim_hidden', type=int, default=50, help='number of units in hidden layer')
    parser.add_argument('--epoch_modulo', type=int, default=2000, help='number of epochs between logging results')
    parser.add_argument('--epochs', type=int, default=30000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed_data', type=int, default=112358, help='seed to generate data')
    parser.add_argument('--seed_init', type=int, default=853211, help='seed to initialize model')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'convergence')
    os.makedirs(exp_path, exist_ok=True)

    # generate data
    tf.keras.utils.set_random_seed(args.seed_data)
    data = generate_toy_data()
    with open(os.path.join(exp_path, 'data.pkl'), 'wb') as f:
        pickle.dump(data, f)

    # data dimension
    dim_x = data['x_train'].shape[1]
    dim_y = data['y_train'].shape[1]

    # initial network weights
    tf.keras.utils.set_random_seed(args.seed_init)
    f_hidden_layer_init = f_hidden_layer(dim_x, args.dim_hidden).get_weights()
    f_mean_output_layer_init = f_output_layer(args.dim_hidden, dim_y, f_out=None).get_weights()
    f_scale_output_layer_init = f_output_layer(args.dim_hidden, dim_y, f_out='softplus').get_weights()

    # loop over models
    metrics = pd.DataFrame()
    measurements = pd.DataFrame()
    for i, mdl in enumerate([UnitVariance, Heteroscedastic, SecondOrderMean, FaithfulHeteroscedastic]):

        # loop over architectures
        for architecture in (['single'] if i == 0 else ['separate', 'shared']):

            # network configuration
            if architecture in {'single', 'shared'}:
                f_trunk = f_hidden_layer
                f_param = f_output_layer
            else:
                f_trunk = None
                f_param = f_neural_net

            # initialize and compile model such that all models share a common initialization
            model = mdl(dim_x, dim_y, f_param, f_trunk, d_hidden=args.dim_hidden)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])
            if architecture in {'single', 'shared'}:
                model.f_trunk.set_weights(f_hidden_layer_init)
                model.f_mean.set_weights(f_mean_output_layer_init)
                if hasattr(model, 'f_scale'):
                    model.f_scale.set_weights(f_scale_output_layer_init)
            else:
                model.f_mean.set_weights(f_hidden_layer_init + f_mean_output_layer_init)
                model.f_scale.set_weights(f_hidden_layer_init + f_scale_output_layer_init)

            # index for this model/architecture
            model_name = pretty_model_name(model)
            index = pd.MultiIndex.from_tuples([(model_name, architecture)], names=['Model', 'Architecture'])
            print('********** ' + model_name + ' (' + architecture + ') **********')

            # loop over the epochs
            for epoch in range(0, args.epochs + 1, args.epoch_modulo):
                print('\rEpoch {:d} of {:d}'.format(epoch, args.epochs), end='')

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

            # new line
            print('')

    # save performance measures
    metrics.to_pickle(os.path.join(exp_path, 'metrics.pkl'))
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
