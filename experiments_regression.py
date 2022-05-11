import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from metrics import MeanLogLikelihood, RootMeanSquaredError, ExpectationCalibrationError
from sklearn import preprocessing
from regression_data import create_or_load_fold
from regression_models import Normal, Student, VariationalGammaNormal

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boston', help='which dataset to use')
parser.add_argument('--num_folds', type=int, default=10, help='number of folds')
parser.add_argument('--num_trials', type=int, default=10, help='number of trials per fold')
parser.add_argument('--split_seed', type=int, default=853211, help='number of trials per fold')
parser.add_argument('--trial_seed', type=int, default=112358, help='number of trials per fold')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace existing results')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'regression', args.dataset)
os.makedirs(exp_path, exist_ok=True)

# models and configurations to run
nn_kwargs = {'n_hidden': 2, 'd_hidden': 50, 'f_hidden': 'elu'}
models_and_configs = [
    {'model': Normal, 'config': {'optimization': 'first-order'}},
    {'model': Normal, 'config': {'optimization': 'second-order-mean'}},
    {'model': Student, 'config': {'optimization': 'second-order-mean'}},
    {'model': VariationalGammaNormal, 'config': {'empirical_bayes': False}},
    {'model': VariationalGammaNormal, 'config': {'empirical_bayes': True, 'sq_err_scale': 0.6}},
    {'model': VariationalGammaNormal, 'config': {'empirical_bayes': True, 'sq_err_scale': 0.8}},
    {'model': VariationalGammaNormal, 'config': {'empirical_bayes': True, 'sq_err_scale': 1.0}},
]

# create or load folds
np.random.seed(args.split_seed)
data = create_or_load_fold(args.dataset, args.num_folds, exp_path)

# loop over folds
results = pd.DataFrame()
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

        # loop over the models and configurations
        for model_and_config in models_and_configs:
            print('\n********* Fold {:d} | Trial {:d} *********'.format(fold, trial))

            # each trial within a fold gets the same random seed
            tf.keras.utils.set_random_seed(trial * args.trial_seed)

            # configure and build model
            model = model_and_config['model'](
                dim_x=x_train.shape[1],
                dim_y=y_train.shape[1],
                y_mean=tf.reduce_mean(y_train, axis=0),
                y_var=tf.math.reduce_variance(y_train, axis=0),
                **model_and_config['config'], **nn_kwargs
            )
            optimizer = tf.keras.optimizers.Adam(1e-3)
            model.compile(optimizer=optimizer, metrics=[
                MeanLogLikelihood(),
                RootMeanSquaredError(),
                ExpectationCalibrationError()
            ])

            # determine directory where to save model
            model_dir = os.path.join(trial_path, model.name)

            # if we are set to resume and the model directory already contains a saved model, load it
            if not bool(args.replace) and os.path.exists(os.path.join(model_dir, 'checkpoint')):
                print(model.name + ' exists. Loading...')
                model.load_weights(os.path.join(model_dir, 'best_checkpoint'))

            # otherwise, train and save the model
            else:
                valid_freq = 100
                hist = model.fit(x=x_train, y=y_train,
                                 validation_data=(x_valid, y_valid), validation_freq=valid_freq,
                                 batch_size=x_train.shape[0], epochs=int(20e3), verbose=0,
                                 callbacks=[RegressionCallback(validation_freq=valid_freq, early_stop_patience=10)])
                model.save_weights(os.path.join(model_dir, 'best_checkpoint'))
                with open(os.path.join(model_dir, 'hist.pkl'), 'wb') as f:
                    pickle.dump(hist.history, f)

            # test model
            test_metrics = model.evaluate(x=x_valid, y=y_valid, verbose=0)
            print('Test LL = {:.4f} | Test RMSE = {:.4f} | Test ECE = {:.4f}'.format(*test_metrics))

            # update results table
            new_results = pd.DataFrame(
                data={'Model': model.name, 'Architecture': str(nn_kwargs),
                      'LL': test_metrics[0], 'RMSE': test_metrics[1], 'ECE': test_metrics[2]},
                index=pd.MultiIndex.from_arrays([[fold], [trial]], names=['fold', 'trial']))
            results = pd.concat([results, new_results])

# save the results
results.to_pickle(os.path.join(exp_path, 'results.pkl'))
