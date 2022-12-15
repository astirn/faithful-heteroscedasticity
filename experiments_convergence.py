import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from datasets import generate_toy_data
from metrics import RootMeanSquaredError
from models import f_hidden_layers, f_output_layer, get_models_and_configurations
from utils import model_config_index, pretty_model_name

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch_modulo', type=int, default=2000, help='number of epochs between logging results')
parser.add_argument('--epochs', type=int, default=20000, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace saved model')
parser.add_argument('--seed', type=int, default=112358, help='random number seed for reproducibility')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'convergence')
os.makedirs(exp_path, exist_ok=True)

# enable GPU determinism
tf.config.experimental.enable_op_determinism()

# generate data
tf.keras.utils.set_random_seed(args.seed)
data = generate_toy_data()
with open(os.path.join(exp_path, 'data.pkl'), 'wb') as f:
    pickle.dump(data, f)

# data dimension
dim_x = data['x_train'].shape[1]
dim_y = data['y_train'].shape[1]

# models, architectures and configurations to run
models_and_configurations = get_models_and_configurations(
    nn_kwargs=dict(f_trunk=f_hidden_layers, f_param=f_output_layer, d_hidden=(50,)),
    mcd_kwargs=dict(mc_samples=250), de_kwargs=dict(), student_kwargs=dict())

# initialize/load optimization history
opti_history_file = os.path.join(exp_path, 'optimization_history.pkl')
opti_history = pd.read_pickle(opti_history_file) if os.path.exists(opti_history_file) else pd.DataFrame()
measurements_file = os.path.join(exp_path, 'measurements.pkl')
measurements = pd.read_pickle(measurements_file) if os.path.exists(measurements_file) else pd.DataFrame()

# loop over models/architectures/configurations
for mag in models_and_configurations:

    # model configuration (seed and GPU determinism ensures architectures are identically initialized)
    tf.keras.utils.set_random_seed(args.seed)
    model = mag['model'](dim_x=dim_x, dim_y=dim_y, **mag['model_kwargs'], **mag['nn_kwargs'])
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), metrics=[RootMeanSquaredError()])

    # index for this model and configuration
    model_name = pretty_model_name(model, mag['model_kwargs'])
    index, index_str = model_config_index(model_name, model.model_class, **mag['nn_kwargs'])
    print('***** {:s} *****'.format(index_str))

    # if results exist, continue unless we are forcing their replacement
    if not bool(args.replace) and index.isin(opti_history.index) and index.isin(measurements.index):
        print('Results exist. Continuing...')
        continue

    # if proceeding with new results, make sure any existing results are cleared out
    if index.isin(opti_history.index):
        opti_history.drop(index, inplace=True)
    if index.isin(measurements.index):
        measurements.drop(index, inplace=True)

    # loop over the epochs
    for epoch in range(0, args.epochs + 1, args.epoch_modulo):
        print('\rEpoch {:d} of {:d}'.format(epoch, args.epochs), end='')

        # improve fit by specified number of epochs
        if epoch > 0:
            hist = model.fit(x=data['x_train'], y=data['y_train'],
                             batch_size=data['x_train'].shape[0], epochs=args.epoch_modulo, verbose=0)
            opti_history = pd.concat([opti_history, pd.DataFrame(
                data={'Epoch': epoch - args.epoch_modulo + np.array(hist.epoch), 'RMSE': hist.history['RMSE']},
                index=index.repeat(args.epoch_modulo))])

        # measure performance
        py_x = model.predictive_distribution(x=data['x_test'])
        measurements = pd.concat([measurements, pd.DataFrame(
            data={'Epoch': epoch, 'x': data['x_test'][:, 0],
                  'Mean': py_x.mean()[:, 0], 'Std. Deviation': py_x.stddev()[:, 0]},
            index=index.repeat(data['x_test'].shape[0]))])

    # save performance measures
    print('')
    opti_history.to_pickle(opti_history_file)
    measurements.to_pickle(measurements_file)
