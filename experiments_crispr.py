import argparse
import os
import pickle

import models_regression as models
import numpy as np
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from layers import SHAPyCat
from metrics import RootMeanSquaredError, ExpectedCalibrationError
from shap import DeepExplainer


# sequence representation network
def f_trunk(d_in):
    return tf.keras.Sequential(name='SequenceTrunk', layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
        tf.keras.layers.Flatten()])


# parameter network
def f_param(d_in, **kwargs):
    return models.param_net(d_in=d_in, d_out=1, d_hidden=[128, 32])


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dataset', type=str, default='junction', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--num_folds', type=int, default=10, help='number of pre-validation folds')
    parser.add_argument('--seed', type=int, default=12345, help='number of trials per fold')
    parser.add_argument('--replace', action='store_true', default=False, help='whether to replace existing results')
    args = parser.parse_args()

    # make experimental directory base path etc...
    exp_path = os.path.join('experiments', 'crispr', args.dataset)
    os.makedirs(exp_path, exist_ok=True)
    folds_file = os.path.join(exp_path, 'folds.npy')

    # load data
    with open(os.path.join('data', 'crispr', args.dataset + '.pkl'), 'rb') as f:
        x, y, nt_lut = pickle.load(f).values()
    x = tf.one_hot(x, depth=4)
    y = tf.expand_dims(y, axis=1)

    # create or load fold assignments
    np.random.seed(args.seed)
    if os.path.exists(folds_file):
        folds = np.load(folds_file)
    else:
        folds = np.random.choice(args.num_folds, size=x.shape[0]) + 1
        np.save(folds_file, folds)

    # loop over validation folds
    measurements = pd.DataFrame()
    shap = pd.DataFrame()
    for k in range(1, args.num_folds + 1):
        print('******************** Fold {:d}/{:d} ********************'.format(k, args.num_folds))
        fold_path = os.path.join(exp_path, 'fold_' + str(k))
        i_train, i_valid = tf.not_equal(folds, k), tf.equal(folds, k)

        # loop over models
        for mdl in [models.UnitVarianceNormal, models.HeteroscedasticNormal, models.FaithfulHeteroscedasticNormal]:

            # initialize model
            tf.keras.utils.set_random_seed(k * args.seed)
            model = mdl(dim_x=x.shape[1:], dim_y=1, f_param=f_param, f_trunk=f_trunk)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          run_eagerly=args.debug,
                          metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

            # determine where to save model
            save_path = os.path.join(fold_path, model.name)

            # if we are set to resume and the model directory already contains a saved model, load it
            if not bool(args.replace) and os.path.exists(os.path.join(save_path, 'checkpoint')):
                print(model.name + ' exists. Loading...')
                checkpoint = tf.train.Checkpoint(model)
                checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()

            # otherwise, train and save the model
            else:
                hist = model.fit(x=x[i_train], y=y[i_train], validation_data=(x[i_valid], y[i_valid]),
                                 batch_size=args.batch_size, epochs=int(10e3), verbose=0,
                                 callbacks=[RegressionCallback(early_stop_patience=100)])
                model.save_weights(os.path.join(save_path, 'best_checkpoint'))

            # index for this model
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = pd.Index([model_name], name='Model')

            # save local performance measurements
            params = model.predict(x=x[i_valid], verbose=0)
            squared_errors = tf.reduce_sum((y[i_valid] - params['mean']) ** 2, axis=-1)
            cdf_y = tf.reduce_sum(model.predictive_distribution(**params).cdf(y[i_valid]), axis=-1)
            measurements = pd.concat([measurements, pd.DataFrame({
                'squared errors': squared_errors,
                'F(y)': cdf_y,
            }, index.repeat(len(y[i_valid])))])

            # convert model into a Sequential model because the SHAP package does not support otherwise
            shapy_cat = tf.keras.Sequential(layers=[tf.keras.layers.InputLayer(x.shape[1:]), SHAPyCat(model)])
            shapy_cat_params = tf.split(shapy_cat(x[i_valid]), num_or_size_splits=2, axis=-1)
            for i, key in enumerate(params.keys()):
                min_abs_error = tf.reduce_min(tf.abs(shapy_cat_params[i] - params[key])).numpy()
                assert min_abs_error == 0.0, 'bad SHAPy cat!'

            # compute SHAP values
            e = DeepExplainer(shapy_cat, tf.random.shuffle(x[i_train])[:min(5000, x[i_train].shape[0])].numpy())
            shap_values = e.shap_values(x[i_valid].numpy())
            # shap = pd.concat([shap, pd.DataFrame({
            #     'squared errors': squared_errors,
            #     'F(y)': cdf_y,
            # }, index.repeat(x[i_valid].shape[0]))])

            # clear out memory
            tf.keras.backend.clear_session()

    # save performance measures and SHAP values
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
    shap.to_pickle(os.path.join(exp_path, 'shap.pkl'))
