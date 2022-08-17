import argparse
import os
import pickle

import models_regression as models
import numpy as np
import pandas as pd
import tensorflow as tf

from data_regression import generate_toy_data
from metrics import RootMeanSquaredError, ExpectedCalibrationError

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--epoch_modulo', type=int, default=1000, help='number of epochs beteween logging results')
parser.add_argument('--epochs', type=int, default=20000, help='number of training epochs')
parser.add_argument('--learning_rate', type=float, default=5e-3, help='learning rate')
parser.add_argument('--seed', type=int, default=12345, help='number of trials per fold')
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
mean_model = models.UnitVarianceNormal(data['x_train'].shape[1], data['y_train'].shape[1], models.param_net)
mean_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                   metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

# initialize heteroscedastic model such that it starts with the same mean network initialization
full_model = models.HeteroscedasticNormal(data['x_train'].shape[1], data['y_train'].shape[1], models.param_net)
full_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                   metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])
full_model.f_mean.set_weights(mean_model.get_weights())

# initialize faithful heteroscedastic model such that it starts with the same mean network initialization
faith_model = models.FaithfulHeteroscedasticNormal(data['x_train'].shape[1], data['y_train'].shape[1], models.param_net)
faith_model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                    metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])
faith_model.f_mean.set_weights(mean_model.get_weights())

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
                data={'Epoch': epoch + np.array(hist.epoch), 'RMSE': hist.history['RMSE'], 'ECE': hist.history['ECE']},
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
