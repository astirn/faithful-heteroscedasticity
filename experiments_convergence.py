import argparse
import os
import pickle

import models as mdl
import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import generate_toy_data
from metrics import RootMeanSquaredError, ExpectedCalibrationError
from utils import pretty_model_name

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dim_hidden', type=int, default=50, help='number of units in hidden layer')
parser.add_argument('--epoch_modulo', type=int, default=2000, help='number of epochs between logging results')
parser.add_argument('--epochs', type=int, default=30000, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=112358, help='random number seed for reproducibility')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'convergence')
os.makedirs(exp_path, exist_ok=True)

# set random seed
tf.keras.utils.set_random_seed(args.seed)

# generate data
data = generate_toy_data()
with open(os.path.join(exp_path, 'data.pkl'), 'wb') as f:
    pickle.dump(data, f)

# data dimension
dim_x = data['x_train'].shape[1]
dim_y = data['y_train'].shape[1]

# initial network weights
d_hidden = [args.dim_hidden]
f_hidden_layer_init = mdl.f_hidden_layers(dim_x, d_hidden).get_weights()
f_mean_output_layer_init = mdl.f_output_layer(d_hidden[-1], dim_y, f_out=None).get_weights()
f_scale_output_layer_init = mdl.f_output_layer(d_hidden[-1], dim_y, f_out='softplus').get_weights()

# loop over models
metrics = pd.DataFrame()
measurements = pd.DataFrame()
for i, model in enumerate([mdl.UnitVariance, mdl.Heteroscedastic, mdl.SecondOrderMean, mdl.FaithfulHeteroscedastic]):

    # loop over architectures
    for architecture in (['single'] if i == 0 else ['separate', 'shared']):

        # network configuration
        if architecture in {'single', 'shared'}:
            f_trunk = mdl.f_hidden_layers
            f_param = mdl.f_output_layer
        else:
            f_trunk = None
            f_param = mdl.f_neural_net

        # initialize and compile model such that all models share a common initialization
        model = model(dim_x, dim_y, f_param, f_trunk, d_hidden=d_hidden)
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
