import argparse
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from datasets import create_or_load_fold
from metrics import RootMeanSquaredError, MeanLogLikelihood
from models import f_hidden_layers, f_output_layer, get_models_and_configurations
from sklearn import preprocessing
from utils import model_config_dir, model_config_index, pretty_model_name, ZScoreNormalization

from tensorflow_probability import distributions as tfd

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boston', help='which dataset to use')
parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
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
models_and_configurations = get_models_and_configurations(
    nn_kwargs=dict(f_trunk=f_hidden_layers, f_param=f_output_layer, d_hidden=(50, 50)),
    mcd_kwargs=dict(), de_kwargs=dict(), student_kwargs=dict())

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
        fold_seed = int(zlib.crc32(str(fold * trial_seed).encode())) % (2 ** 32 - 1)

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

            # model configuration (seed and GPU determinism ensures architectures are identically initialized)
            tf.keras.utils.set_random_seed(fold_seed)
            model = mag['model'](dim_x=dim_x, dim_y=dim_y, **mag['model_kwargs'], **mag['nn_kwargs'])
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          run_eagerly=args.debug,
                          metrics=[RootMeanSquaredError(), MeanLogLikelihood()])

            # index for this model and configuration
            model_name = pretty_model_name(model, mag['model_kwargs'])
            index, index_str = model_config_index(model_name, model.model_class, **mag['nn_kwargs'])
            print('********** Trial {:d} | Fold {:d} | {:s} **********'.format(trial, fold, index_str))

            # determine where to save model
            save_path = model_config_dir(fold_path, model, mag['model_kwargs'], mag['nn_kwargs'])

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
                tf.keras.utils.set_random_seed(fold_seed)
                hist = model.fit(x=x_train, y=z_normalization.normalize_targets(y_train),
                                 validation_data=(x_valid, z_normalization.normalize_targets(y_valid)),
                                 validation_freq=valid_freq, batch_size=x_train.shape[0], epochs=int(60e3), verbose=0,
                                 callbacks=[RegressionCallback(validation_freq=valid_freq, early_stop_patience=100)])
                model.save_weights(os.path.join(save_path, 'best_checkpoint'))
                history = hist.history
                with open(os.path.join(save_path, 'hist.pkl'), 'wb') as f:
                    pickle.dump(history, f)

            # save local performance measurements
            tf.keras.utils.set_random_seed(fold_seed)
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
                    'squared errors': tf.reduce_sum((y - py_x.mean()) ** 2, axis=-1),
                    'z': ((y - py_x.mean()) / py_x.stddev()).numpy().tolist(),
                }, index.repeat(len(y_valid)))])

# save performance measures
measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
