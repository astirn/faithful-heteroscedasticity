import argparse
import os

import numpy as np
import tensorflow as tf

from callbacks import RegressionCallback
from metrics import MeanLogLikelihood, RootMeanSquaredError, ExpectationCalibrationError
from sklearn import preprocessing
from regression_data import create_or_load_fold
from regression_models import Normal, VariationalRegression

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='boston', help='which dataset to use')
parser.add_argument('--num_folds', type=int, default=10, help='number of folds')
parser.add_argument('--num_trials', type=int, default=10, help='number of trials per fold')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace existing results')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'regression', args.dataset)
os.makedirs(exp_path, exist_ok=True)

# models and configurations to run
models_and_configs = [
    {'model': Normal, 'config': {'optimization': 'first-order'}},
    {'model': Normal, 'config': {'optimization': 'second-order-mean'}},
    {'model': VariationalRegression, 'config': dict()},
    # {'model': Normal, 'config': {'optimization': 'second-order-diag'}},
    # {'model': Normal, 'config': {'optimization': 'second-order-full'}},
]

# create or load folds
data = create_or_load_fold(args.dataset, args.num_folds, exp_path)

# loop over folds
for fold in np.unique(data['split']):
    fold_path = os.path.join(exp_path, 'fold_' + str(fold))

    # loop over trials
    for trial in range(1, args.num_trials + 1):
        trial_path = os.path.join(fold_path, 'trial_' + str(fold))

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
            print('********* Fold {:d} | Trial {:d} *********'.format(fold, trial))

            # each trial within a fold gets the same random seed
            tf.keras.utils.set_random_seed(trial)

            # configure and build model
            model = model_and_config['model'](
                dim_x=x_train.shape[1],
                dim_y=y_train.shape[1],
                y_mean=tf.reduce_mean(y_train, axis=0),
                y_var=tf.math.reduce_variance(y_train, axis=0),
                **model_and_config['config']
            )
            optimizer = tf.keras.optimizers.Adam(1e-3)
            model.compile(optimizer=optimizer, metrics=[
                MeanLogLikelihood(),
                RootMeanSquaredError(),
                ExpectationCalibrationError()
            ])

            # train model
            valid_freq = 100
            hist = model.fit(x=x_train, y=y_train,
                             validation_data=(x_valid, y_valid), validation_freq=valid_freq,
                             batch_size=x_train.shape[0], epochs=int(20e3), verbose=0,
                             callbacks=[RegressionCallback(validation_freq=valid_freq, early_stop_patience=10)])

            # test model
            test_metrics = model.evaluate((x_valid, y_valid), verbose=0)
            print('Test LL = {:.4f} | Test RMSE = {:.4f} | Test ECE = {:.4f}'.format(*test_metrics))


        # # determine directory where to save model
        # model_dir = utils.model_save_dir(fold_path, normalization, context, loss, model)
        #
        # # allows some retries since model can infrequently and randomly fail to converge
        # for n in range(args.max_attempts):
        #     if n > 0:
        #         print('Reattempt {:d} after r = {:.2f}'.format(n, r))
        #
        #     # if we are set to resume and the model directory already contains a saved model, load it
        #     if n == 0 and not bool(args.replace) and os.path.exists(os.path.join(model_dir, 'saved_model.pb')):
        #         print('Model exists. Loading...')
        #         model = load_model(model_dir, model_dir.split('/')[-1].split('-')[0])
        #
        #     # otherwise, train and save the model
        #     else:
        #         model = train(model, train_data, valid_data, verbose=0)
        #         model.save(model_dir, save_traces=True)
        #         print('Model saved!')
        #
        #     # break retry loop if the Pearson correlation coefficient is satisfactory
        #     df_tap = normalizer.denormalize(utils.accumulate_targets_and_predictions(model, valid_data))
        #     predictions = {'predicted_lfc': df_tap.get('predicted_lfc'),
        #                    'predicted_label_likelihood': df_tap.get('predicted_label_likelihood')}
        #     r = utils.regression_metrics(df_tap['target_lfc'], **predictions)[0]
        #     if r >= args.retry_threshold:
        #         break
        #
        #     # otherwise, recompile the model and try again
        #     else:
        #         model = build_and_compile(model_name, loss, context, train_data, **args.kwargs)
