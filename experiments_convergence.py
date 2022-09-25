import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import generate_toy_data
from metrics import RootMeanSquaredError, ExpectedCalibrationError
from models import f_hidden_layers, f_output_layer, f_neural_net, get_models_architectures_configurations
from utils import pretty_model_name

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dim_hidden', type=int, default=50, help='number of units in hidden layer')
parser.add_argument('--epoch_modulo', type=int, default=2000, help='number of epochs between logging results')
parser.add_argument('--epochs', type=int, default=30000, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace saved model')
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
f_hidden_layer_init = f_hidden_layers(dim_x, d_hidden).get_weights()
f_mean_output_layer_init = f_output_layer(d_hidden[-1], dim_y, f_out=None).get_weights()
f_scale_output_layer_init = f_output_layer(d_hidden[-1], dim_y, f_out='softplus').get_weights()

# models, architectures and configurations to run
nn_kwargs = dict(d_hidden=d_hidden, f_hidden='elu')
models_architectures_configurations = get_models_architectures_configurations(nn_kwargs)

# initialize/load optimization history
metrics_file = os.path.join(exp_path, 'metrics.pkl')
metrics = pd.read_pickle(metrics_file) if os.path.exists(metrics_file) else pd.DataFrame()
measurements_file = os.path.join(exp_path, 'measurements.pkl')
measurements = pd.read_pickle(measurements_file) if os.path.exists(measurements_file) else pd.DataFrame()

# loop over models
for i, mag in enumerate(models_architectures_configurations):

    # model configuration
    if mag['architecture'] == 'separate':
        model = mag['model'](dim_x=dim_x, dim_y=dim_y, f_trunk=None, f_param=f_neural_net, **mag['config'])
    elif mag['architecture'] in {'single', 'shared'}:
        model = mag['model'](dim_x=dim_x, dim_y=dim_y, f_trunk=f_hidden_layers, f_param=f_output_layer, **mag['config'])
    else:
        raise NotImplementedError

    # initialize model such that all models share a common initialization
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])
    if mag['architecture'] in {'single', 'shared'}:
        model.f_trunk.set_weights(f_hidden_layer_init)
        model.f_mean.set_weights(f_mean_output_layer_init)
        if hasattr(model, 'f_scale'):
            model.f_scale.set_weights(f_scale_output_layer_init)
    else:
        model.f_mean.set_weights(f_hidden_layer_init + f_mean_output_layer_init)
        model.f_scale.set_weights(f_hidden_layer_init + f_scale_output_layer_init)

    # index for this model/architecture
    model_name = pretty_model_name(model)
    index = pd.MultiIndex.from_tuples([(model_name, mag['architecture'])], names=['Model', 'Architecture'])
    print('********** ' + model_name + ' (' + mag['architecture'] + ') **********')

    # if results exist, continue unless we are forcing their replacement
    if not bool(args.replace) and index.isin(metrics.index) and index.isin(measurements.index):
        print('Results exist. Continuing...')
        continue

    # if proceeding with new results, make sure any existing results are cleared out
    if index.isin(metrics.index):
        metrics.drop(index, inplace=True)
    if index.isin(measurements.index):
        measurements.drop(index, inplace=True)

    # loop over the epochs
    for epoch in range(0, args.epochs + 1, args.epoch_modulo):
        print('\rEpoch {:d} of {:d}'.format(epoch, args.epochs), end='')

        # improve fit by specified number of epochs
        if epoch > 0:
            hist = model.fit(x=data['x_train'], y=data['y_train'],
                             batch_size=data['x_train'].shape[0], epochs=args.epoch_modulo, verbose=0)
            metrics = pd.concat([metrics, pd.DataFrame(
                data={'Epoch': epoch - args.epoch_modulo + np.array(hist.epoch),
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
    print('')
    metrics.to_pickle(os.path.join(exp_path, 'metrics.pkl'))
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
