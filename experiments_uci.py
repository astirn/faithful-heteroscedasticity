import argparse
import json
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from datasets import create_or_load_fold
from metrics import RootMeanSquaredError
from models import f_hidden_layers, f_output_layer, f_neural_net, get_models_and_configurations
from sklearn import preprocessing
from utils import pretty_model_name, ZScoreNormalization

from tensorflow_probability import distributions as tfd

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boston', help='which dataset to use')
parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
parser.add_argument('--num_folds', type=int, default=10, help='number of folds')
parser.add_argument('--num_trials', type=int, default=1, help='number of trials per fold')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace saved model')
parser.add_argument('--seed', type=int, default=112358, help='random number seed for reproducibility')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'uci', args.dataset)
os.makedirs(exp_path, exist_ok=True)

# enable GPU determinism
tf.config.experimental.enable_op_determinism()

# models and configurations to run
models_and_configurations = get_models_and_configurations(dict(d_hidden=(50, 50), f_hidden='elu'))

# loop over trials
measurements = pd.DataFrame()
for trial in range(1, args.num_trials + 1):
    trial_path = os.path.join(exp_path, 'trial_' + str(trial))

    # a deterministic but seemingly random transformation of the provided seed into a trial seed
    trial_seed = int(zlib.crc32(str(trial * args.seed).encode())) % (2 ** 32 - 1)

    # create or load folds for this trial
    tf.keras.utils.set_random_seed(trial_seed)
    data = create_or_load_fold(args.dataset, args.num_folds, save_path=trial_path)
    dim_x = data['covariates'].shape[1]
    dim_y = data['response'].shape[1]

    # loop over folds
    for fold in np.unique(data['split']):
        fold_path = os.path.join(trial_path, 'fold_' + str(fold))

        # a deterministic but seemingly random transformation of the trial seed into a fold seed
        fold_seed = int(zlib.crc32(str(trial * trial_seed).encode())) % (2 ** 32 - 1)

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

        # loop over models/architectures/configurations
        for mag in models_and_configurations:
            print('\n********* Fold {:d} | Trial {:d} *********'.format(fold, trial))

            # model configuration (seed and GPU determinism ensures architectures are identically initialized)
            tf.keras.utils.set_random_seed(fold_seed)
            if mag['architecture'] == 'separate':
                model = mag['model'](dim_x=x_train.shape[1], dim_y=y_train.shape[1],
                                     f_trunk=None, f_param=f_neural_net,
                                     **mag['config'], **mag['nn_kwargs'])
            elif mag['architecture'] in {'single', 'shared'}:
                model = mag['model'](dim_x=x_train.shape[1], dim_y=y_train.shape[1],
                                     f_trunk=f_hidden_layers, f_param=f_output_layer,
                                     **mag['config'], **mag['nn_kwargs'])
            else:
                raise NotImplementedError
            optimizer = tf.keras.optimizers.Adam(1e-3)
            model.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[RootMeanSquaredError()])

            # determine where to save model
            config_dir = str(zlib.crc32(json.dumps(mag['nn_kwargs']).encode('utf-8')))
            save_path = os.path.join(fold_path, ''.join([model.name, mag['architecture'].capitalize()]), config_dir)

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
            index = {'Model': pretty_model_name(model), 'Architecture': mag['architecture']}
            index.update(mag['nn_kwargs'])
            index = pd.MultiIndex.from_tuples([tuple(index.values())], names=list(index.keys()))

            # save local performance measurements
            params = model.predict(x=x_valid, verbose=0)
            for normalized in [True, False]:  # True must run first
                if normalized:
                    y = z_normalization.normalize_targets(y_valid)
                else:
                    y = y_valid
                    params = {key: z_normalization.scale_parameters(key, values) for key, values in params.items()}
                py_x = tfd.Independent(model.predictive_distribution(**params), reinterpreted_batch_ndims=1)
                measurements = pd.concat([measurements, pd.DataFrame({
                    'normalized': normalized,
                    'log p(y|x)': py_x.log_prob(y),
                    'squared errors': tf.reduce_sum((y - params['mean']) ** 2, axis=-1),
                    'z': ((y - params['mean']) / params['std']).numpy().tolist(),
                }, index.repeat(len(y_valid)))])

# save performance measures
measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
