import argparse
import itertools
import models
import os
import pickle

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
    parser.add_argument('--dataset', type=str, default='junction-targets', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_background', type=int, default=20000, help='maximum number of SHAP background samples')
    parser.add_argument('--num_folds', type=int, default=10, help='number of pre-validation folds')
    parser.add_argument('--replace', action='store_true', default=False, help='whether to replace existing results')
    parser.add_argument('--seed_data', type=int, default=112358, help='seed to generate folds')
    parser.add_argument('--seed_init', type=int, default=853211, help='seed to initialize model')
    args = parser.parse_args()

    # make experimental directory base path etc...
    exp_path = os.path.join('experiments', 'crispr', args.dataset)
    os.makedirs(exp_path, exist_ok=True)
    folds_file = os.path.join(exp_path, 'folds.npy')

    # load data
    with open(os.path.join('data', 'crispr', args.dataset + '.pkl'), 'rb') as f:
        x, y_mean, y_replicates, sequence = pickle.load(f).values()
    x = tf.one_hot(x, depth=4)

    # create or load fold assignments
    np.random.seed(args.seed_data)
    if os.path.exists(folds_file):
        folds = np.load(folds_file)
    else:
        folds = np.random.choice(args.num_folds, size=x.shape[0]) + 1
        np.save(folds_file, folds)

    # loop over validation folds
    metrics = pd.DataFrame()
    measurements = pd.DataFrame()
    shap = pd.DataFrame()
    for k, observation in itertools.product(range(1, args.num_folds + 1), ['mean', 'replicates']):
        print('******************** Fold {:d}/{:d}: {:s} ********************'.format(k, args.num_folds, observation))
        fold_path = os.path.join(exp_path, 'fold_' + str(k))
        i_train, i_valid = tf.not_equal(folds, k), tf.equal(folds, k)

        # prepare training/validation data according to observation type
        if observation == 'mean':
            x_train, y_train = x[i_train], y_mean[i_train]
            x_valid, y_valid = x[i_valid], y_mean[i_valid]
        elif observation == 'replicates':
            repeats = tf.shape(y_replicates)[1]
            x_train, y_train = tf.repeat(x[i_train], repeats, axis=0), tf.reshape(y_replicates[i_train], [-1, 1])
            x_valid, y_valid = tf.repeat(x[i_valid], repeats, axis=0), tf.reshape(y_replicates[i_valid], [-1, 1])
            i_no_nan = tf.squeeze(~tf.math.is_nan(y_train))
            x_train, y_train = x_train[i_no_nan], y_train[i_no_nan]
            i_no_nan = tf.squeeze(~tf.math.is_nan(y_valid))
            x_valid, y_valid = x_valid[i_no_nan], y_valid[i_no_nan]
            i_train_shuffle = tf.random.shuffle(tf.range(tf.shape(x_train)[0]), seed=k * args.seed_data)
            x_train, y_train = tf.gather(x_train, i_train_shuffle), tf.gather(y_train, i_train_shuffle)
        else:
            raise NotImplementedError

        # loop over models
        for mdl in [models.UnitVariance, models.Heteroscedastic, models.FaithfulHeteroscedastic]:

            # initialize model
            tf.keras.utils.set_random_seed(k * args.seed_init)
            model = mdl(dim_x=x.shape[1:], dim_y=1, f_param=f_param, f_trunk=f_trunk)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          run_eagerly=args.debug,
                          metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

            # index for this model and observation type
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = pd.MultiIndex.from_tuples([(model_name, observation)], names=['Model', 'Observation'])

            # determine where to save model
            save_path = os.path.join(fold_path, model.name, observation)

            # if we are set to resume and the model directory already contains a saved model, load it
            if not bool(args.replace) and os.path.exists(os.path.join(save_path, 'checkpoint')):
                print(model.name + ' exists. Loading...')
                checkpoint = tf.train.Checkpoint(model)
                checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()

            # otherwise, train and save the model
            else:
                hist = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid),
                                 batch_size=args.batch_size, epochs=int(10e3), verbose=0,
                                 callbacks=[RegressionCallback(early_stop_patience=100)])
                model.save_weights(os.path.join(save_path, 'best_checkpoint'))
                metrics = pd.concat([metrics, pd.DataFrame(
                    data={'Fold': k * np.ones(len(hist.epoch), dtype=int),
                          'Epoch': np.array(hist.epoch),
                          'RMSE': hist.history['RMSE'],
                          'ECE': hist.history['ECE']},
                    index=index.repeat(len(hist.epoch))).set_index('Fold', append=True)])

            # save local performance measurements
            params = model.predict(x=x_valid, verbose=0)
            squared_errors = tf.reduce_sum((y_valid - params['mean']) ** 2, axis=-1)
            cdf_y = tf.reduce_sum(model.predictive_distribution(**params).cdf(y_valid), axis=-1)
            measurements = pd.concat([measurements, pd.DataFrame({
                'squared errors': squared_errors,
                'F(y)': cdf_y,
            }, index.repeat(len(y_valid)))])

            # copy model into a Sequential model because the SHAP package does not support otherwise
            shapy_cat = tf.keras.Sequential(layers=[tf.keras.layers.InputLayer(x.shape[1:]), SHAPyCat(model)])
            shapy_cat_params = np.split(shapy_cat.predict(x=x[i_valid], verbose=0), 2, axis=-1)
            if observation == 'mean':
                for i, key in enumerate(params.keys()):
                    max_abs_error = tf.reduce_max(tf.abs(shapy_cat_params[i] - params[key])).numpy()
                    assert max_abs_error == 0.0, 'bad SHAPy cat!'

            # compute SHAP values
            num_background_samples = min(args.max_background, x[i_train].shape[0])
            e = DeepExplainer(shapy_cat, tf.random.shuffle(x[i_train])[:num_background_samples].numpy())
            shap_values = e.shap_values(x[i_valid].numpy())
            shap_dict = dict(sequence=sequence[i_valid].numpy().tolist(),
                             mean=shap_values[0].sum(-1).tolist(),
                             std=shap_values[1].sum(-1).tolist())
            shap = pd.concat([shap, pd.DataFrame(shap_dict, index.repeat(x[i_valid].shape[0]))])

            # clear out memory
            tf.keras.backend.clear_session()

    # convert all TensorFlow strings to Python strings
    shap['sequence'] = shap['sequence'].apply(lambda seq: seq.decode('utf-8'))

    # save performance measures and SHAP values
    metrics.to_pickle(os.path.join(exp_path, 'metrics.pkl'))
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
    shap.to_pickle(os.path.join(exp_path, 'shap.pkl'))
