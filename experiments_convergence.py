import argparse
import os
import pickle

import models_regression as models
import numpy as np
import pandas as pd
import tensorflow as tf

from data_regression import generate_toy_data
from metrics import RootMeanSquaredError

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
parser.add_argument('--epoch_modulo', type=int, default=1000, help='number of epochs beteween logging results')
parser.add_argument('--epochs', type=int, default=20000, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='learning rate')
parser.add_argument('--seed', type=int, default=853211, help='number of trials per fold')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'convergence')
os.makedirs(exp_path, exist_ok=True)

# set random seed
np.random.seed(args.seed)
tf.keras.utils.set_random_seed(args.seed)

# generate data
data = generate_toy_data()
with open(os.path.join(exp_path, 'data.pkl'), 'wb') as f:
    pickle.dump(data, f)

# initialize mean model
mean_model = models.UnitVarianceNormal(dim_x=data['x_train'].shape[1], dim_y=data['y_train'].shape[1])
mean_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), metrics=[RootMeanSquaredError()])

# initialize heteroscedastic model such that it starts with the same mean network initialization
full_model = models.HeteroscedasticNormal(dim_x=data['x_train'].shape[1], dim_y=data['y_train'].shape[1])
full_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), metrics=[RootMeanSquaredError()])
full_model.f_mean.set_weights(mean_model.get_weights())

# initialize faithful heteroscedastic model such that it starts with the same mean network initialization
faith_model = models.FaithfulHeteroscedasticNormal(dim_x=data['x_train'].shape[1], dim_y=data['y_train'].shape[1])
faith_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), metrics=[RootMeanSquaredError()])
faith_model.f_mean.set_weights(mean_model.get_weights())

# loop over the
measurements = pd.DataFrame()
for iteration in range(args.epochs // args.epoch_modulo):
    epoch = iteration * args.epoch_modulo
    print('\rEpoch {:d} of {:d}'.format(epoch, args.epochs), end='')

    # loop over the models
    for model in [mean_model, full_model, faith_model]:
        model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()

        # measure performance
        py_x = model.predictive_distribution(x=data['x_test'])
        index = pd.MultiIndex.from_tuples([(model_name, epoch)], names=['Model', 'Epoch'])
        measurements = pd.concat([measurements, pd.DataFrame(
            data={'x': data['x_test'][:, 0], 'Mean': py_x.mean()[:, 0], 'Std. Deviation': py_x.stddev()[:, 0]},
            index=index.repeat(data['x_test'].shape[0]))])

        # improve fit by specified number of epochs
        hist = model.fit(x=data['x_train'], y=data['y_train'],
                         batch_size=data['x_train'].shape[0], epochs=args.epoch_modulo, verbose=0)


# save performance measures
measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
