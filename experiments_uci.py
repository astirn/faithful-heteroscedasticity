import argparse
import json
import models
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from datasets import create_or_load_fold
from metrics import RootMeanSquaredError
from sklearn import preprocessing
from utils import ZScoreNormalization

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boston', help='which dataset to use')
parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
parser.add_argument('--num_folds', type=int, default=10, help='number of folds')
parser.add_argument('--num_trials', type=int, default=1, help='number of trials per fold')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace existing results')
parser.add_argument('--seed_data', type=int, default=112358, help='seed to generate data')
parser.add_argument('--seed_init', type=int, default=853211, help='seed to initialize model')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'uci', args.dataset)
os.makedirs(exp_path, exist_ok=True)

# models and configurations to run
nn_kwargs_1 = {'d_hidden': (50,), 'f_hidden': 'elu'}
nn_kwargs_2 = {'d_hidden': (50, 50), 'f_hidden': 'elu'}
models_and_configs = [
    {'model': models.UnitVariance, 'config': {**dict(), **nn_kwargs_1}},
    {'model': models.UnitVariance, 'config': {**dict(), **nn_kwargs_2}},
    {'model': models.Heteroscedastic, 'config': {**dict(), **nn_kwargs_1}},
    {'model': models.Heteroscedastic, 'config': {**dict(), **nn_kwargs_2}},
    {'model': models.FaithfulHeteroscedastic, 'config': {**dict(), **nn_kwargs_1}},
    {'model': models.FaithfulHeteroscedastic, 'config': {**dict(), **nn_kwargs_2}},
]

# create or load folds
data = create_or_load_fold(args.dataset, args.num_folds, save_path=exp_path, seed=args.seed_data)

# loop over folds
measurements = pd.DataFrame()
for fold in np.unique(data['split']):
    fold_path = os.path.join(exp_path, 'fold_' + str(fold))

    # loop over trials
    for trial in range(1, args.num_trials + 1):
        trial_path = os.path.join(fold_path, 'trial_' + str(trial))

        # data pipeline
        i_train = data['split'] != fold
        i_valid = data['split'] == fold
        x_train, y_train = data['covariates'][i_train], tf.constant(data['response'][i_train])
        x_valid, y_valid = data['covariates'][i_valid], tf.constant(data['response'][i_valid])
        x_scale = preprocessing.StandardScaler().fit(x_train)
        x_train = tf.constant(x_scale.transform(x_train))
        x_valid = tf.constant(x_scale.transform(x_valid))

        # target and parameter normalization object
        z_normalization = ZScoreNormalization(y_mean=tf.reduce_mean(y_train, axis=0),
                                              y_var=tf.math.reduce_variance(y_train, axis=0))

        # loop over the models and configurations
        for model_and_config in models_and_configs:
            print('\n********* Fold {:d} | Trial {:d} *********'.format(fold, trial))

            # every model within a trial/fold gets the same seed
            tf.keras.utils.set_random_seed(int(zlib.crc32(str(trial * args.seed_init).encode()) * fold) % (2 ** 32 - 1))

            # configure and build model
            model = model_and_config['model'](
                dim_x=x_train.shape[1],
                dim_y=y_train.shape[1],
                f_param=models.param_net,
                **model_and_config['config']
            )
            optimizer = tf.keras.optimizers.Adam(1e-3)
            model.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[RootMeanSquaredError()])

            # determine where to save model
            config_dir = str(zlib.crc32(json.dumps(model_and_config['config']).encode('utf-8')))
            save_path = os.path.join(trial_path, model.name, config_dir)

            # if we are set to resume and the model directory already contains a saved model, load it
            if not bool(args.replace) and os.path.exists(os.path.join(save_path, 'checkpoint')):
                print(model.name + ' exists. Loading...')
                checkpoint = tf.train.Checkpoint(model)
                checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()
                with open(os.path.join(save_path, 'hist.pkl'), 'rb') as f:
                    history = pickle.load(f)

            # otherwise, train and save the model
            else:
                valid_freq = 10
                hist = model.fit(x=x_train, y=z_normalization.normalize_targets(y_train),
                                 validation_data=(x_valid, z_normalization.normalize_targets(y_valid)),
                                 validation_freq=valid_freq, batch_size=x_train.shape[0], epochs=int(50e3), verbose=0,
                                 callbacks=[RegressionCallback(validation_freq=valid_freq, early_stop_patience=100)])
                model.save_weights(os.path.join(save_path, 'best_checkpoint'))
                history = hist.history
                with open(os.path.join(save_path, 'hist.pkl'), 'wb') as f:
                    pickle.dump(history, f)

            # index for this model and configuration
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = {'Model': model_name}
            index.update(model_and_config['config'])
            index = pd.MultiIndex.from_tuples([tuple(index.values())], names=list(index.keys()))

            # save local performance measurements
            params = model.predict(x=x_valid, verbose=0)
            for normalized in [True, False]:  # True must run first
                if normalized:
                    y = z_normalization.normalize_targets(y_valid)
                else:
                    y = y_valid
                    params = {key: z_normalization.scale_parameters(key, values) for key, values in params.items()}
                squared_errors = tf.reduce_sum((y - params['mean']) ** 2, axis=-1)
                cdf_y = tf.reduce_sum(model.predictive_distribution(**params).cdf(y), axis=-1)
                measurements = pd.concat([measurements, pd.DataFrame({
                    'normalized': normalized,
                    'squared errors': squared_errors,
                    'F(y)': cdf_y,
                }, index.repeat(len(y_valid)))])

# save performance measures
measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
