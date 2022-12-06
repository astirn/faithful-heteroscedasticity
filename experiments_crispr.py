import argparse
import itertools
import os
import pickle
import zlib

import numpy as np
import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from layers import SHAPyCat
from metrics import RootMeanSquaredError
from models import f_neural_net, get_models_and_configurations
from shap import DeepExplainer
from utils import model_config_dir, model_config_index, pretty_model_name

from tensorflow_probability import distributions as tfd


# sequence representation network
def f_conv_net(d_in, **kwargs):
    return tf.keras.Sequential(name='SequenceTrunk', layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.Conv1D(filters=64, kernel_size=4, activation='relu', padding='same'),
        tf.keras.layers.MaxPool1D(pool_size=2, padding='same'),
        tf.keras.layers.Flatten()])


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dataset', type=str, default='flow-cytometry', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--max_background', type=int, default=10000, help='maximum number of SHAP background samples')
    parser.add_argument('--num_folds', type=int, default=10, help='number of pre-validation folds')
    parser.add_argument('--replace', action='store_true', default=False, help='whether to replace saved model')
    parser.add_argument('--seed', type=int, default=112358, help='random number seed for reproducibility')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'crispr', args.dataset)
    os.makedirs(exp_path, exist_ok=True)

    # enable GPU determinism
    tf.config.experimental.enable_op_determinism()

    # models and configurations to run
    nn_kwargs = dict(f_trunk=f_conv_net, f_param=f_neural_net, d_hidden=(128, 32))
    models_and_configurations = get_models_and_configurations(nn_kwargs)

    # load data
    with open(os.path.join('data', 'crispr', args.dataset + '.pkl'), 'rb') as f:
        x, y_mean, y_replicates, sequence = pickle.load(f).values()
    x = tf.one_hot(x, depth=4)

    # create or load fold assignments
    tf.keras.utils.set_random_seed(args.seed)
    folds_file = os.path.join(exp_path, 'folds.npy')
    if not bool(args.replace) and os.path.exists(folds_file):
        folds = np.load(folds_file)
    else:
        folds = np.random.choice(args.num_folds, size=x.shape[0]) + 1
        np.save(folds_file, folds)

    # initialize/load optimization history
    opti_history_file = os.path.join(exp_path, 'optimization_history.pkl')
    optimization_history = pd.read_pickle(opti_history_file) if os.path.exists(opti_history_file) else pd.DataFrame()

    # initialize/load SHAP values
    mean_file = os.path.join(exp_path, 'mean.pkl')
    mean = pd.read_pickle(mean_file) if os.path.exists(mean_file) else pd.DataFrame()
    shap_file = os.path.join(exp_path, 'shap.pkl')
    shap = pd.read_pickle(shap_file) if os.path.exists(shap_file) else pd.DataFrame()

    # loop over validation folds and observations
    measurements = pd.DataFrame()
    for fold, observations in itertools.product(range(1, args.num_folds + 1), ['means', 'replicates']):
        fold_path = os.path.join(exp_path, 'fold_' + str(fold))
        i_train, i_valid = tf.not_equal(folds, fold), tf.equal(folds, fold)

        # a deterministic but seemingly random transformation of the experiment seed into a fold seed
        fold_seed = int(zlib.crc32(str(fold * args.seed).encode())) % (2 ** 32 - 1)

        # prepare training/validation data according to observation type
        tf.keras.utils.set_random_seed(fold_seed)
        if observations == 'means':
            x_train, y_train = x[i_train], y_mean[i_train]
            x_valid, y_valid = x[i_valid], y_mean[i_valid]
        elif observations == 'replicates':
            repeats = tf.shape(y_replicates)[1]
            x_train, y_train = tf.repeat(x[i_train], repeats, axis=0), tf.reshape(y_replicates[i_train], [-1, 1])
            x_valid, y_valid = tf.repeat(x[i_valid], repeats, axis=0), tf.reshape(y_replicates[i_valid], [-1, 1])
            i_no_nan = tf.squeeze(~tf.math.is_nan(y_train))
            x_train, y_train = x_train[i_no_nan], y_train[i_no_nan]
            i_no_nan = tf.squeeze(~tf.math.is_nan(y_valid))
            x_valid, y_valid = x_valid[i_no_nan], y_valid[i_no_nan]
            i_train_shuffle = tf.random.shuffle(tf.range(tf.shape(x_train)[0]), seed=fold_seed)
            x_train, y_train = tf.gather(x_train, i_train_shuffle), tf.gather(y_train, i_train_shuffle)
        else:
            raise NotImplementedError

        # loop over models/architectures/configurations
        for mag in models_and_configurations:

            # model configuration (seed and GPU determinism ensures architectures are identically initialized)
            tf.keras.utils.set_random_seed(fold_seed)
            model = mag['model'](dim_x=x.shape[1:], dim_y=1, **mag['model_kwargs'], **mag['nn_kwargs'])
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), metrics=[RootMeanSquaredError()])

            # index for this model and configuration
            model_name = pretty_model_name(model, mag['model_kwargs'])
            index, index_str = model_config_index(model_name, model.model_class,
                                                  **{**{'Observations': observations}, **mag['nn_kwargs']})
            print('***** Fold {:d}/{:d}: {:s} *****'.format(fold, args.num_folds, index_str))

            # SHAP index adds fold information
            shap_index = pd.MultiIndex.from_tuples([index.values[0] + (fold,)], names=index.names + ['Fold'])

            # determine where to save model
            save_path = os.path.join(fold_path, observations)
            save_path = model_config_dir(save_path, model, mag['model_kwargs'], mag['nn_kwargs'])

            # if set to resume and a trained model and its optimization history exist, load the existing model
            if not bool(args.replace) \
               and os.path.exists(os.path.join(save_path, 'checkpoint')) \
               and index.isin(optimization_history.index):
                print(model.name + ' exists. Loading...')
                checkpoint = tf.train.Checkpoint(model)
                checkpoint.restore(os.path.join(save_path, 'best_checkpoint')).expect_partial()

            # otherwise, train and save the model
            else:
                tf.keras.utils.set_random_seed(fold_seed)
                hist = model.fit(x=x_train, y=y_train, validation_data=(x_valid, y_valid),
                                 batch_size=args.batch_size, epochs=int(10e3), verbose=0,
                                 callbacks=[RegressionCallback(early_stop_patience=100)])
                model.save_weights(os.path.join(save_path, 'best_checkpoint'))
                if index.isin(optimization_history.index):
                    optimization_history.drop(index, inplace=True)
                if shap_index.isin(mean.index):
                    mean.drop(shap_index, inplace=True)
                if shap_index.isin(shap.index):
                    shap.drop(shap_index, inplace=True)
                optimization_history = pd.concat([optimization_history, pd.DataFrame(
                    data={'Epoch': np.array(hist.epoch), 'RMSE': hist.history['RMSE']},
                    index=index.repeat(len(hist.epoch)))])
                optimization_history.to_pickle(opti_history_file)

            # generate predictions on the validation set
            tf.keras.utils.set_random_seed(fold_seed)
            params = model.predict(x=x_valid, verbose=0)

            # save performance measurements
            py_x = tfd.Independent(model.predictive_distribution(**params), reinterpreted_batch_ndims=1)
            measurements = pd.concat([measurements, pd.DataFrame({
                'log p(y|x)': py_x.log_prob(y_valid),
                'squared errors': tf.reduce_sum((y_valid - py_x.mean()) ** 2, axis=-1),
                'z': ((y_valid - py_x.mean()) / py_x.stddev()).numpy().tolist(),
            }, index.repeat(len(y_valid)))])

            # copy model into a Sequential model because the SHAP package does not support otherwise
            shapy_cat = tf.keras.Sequential(layers=[tf.keras.layers.InputLayer(x.shape[1:]), SHAPyCat(model)])
            shapy_cat_params = np.split(shapy_cat.predict(x=x[i_valid], verbose=0), 2, axis=-1)
            if observations == 'means':
                for j, key in enumerate(params.keys()):
                    max_abs_error = tf.reduce_max(tf.abs(shapy_cat_params[j] - params[key])).numpy()
                    assert max_abs_error == 0.0, 'bad SHAPy cat!'

            # compute SHAP values if we don't have them
            if not shap_index.isin(mean.index) or not shap_index.isin(shap.index):
                tf.keras.utils.set_random_seed(fold_seed)
                num_background_samples = min(args.max_background, x[i_train].shape[0])
                e = DeepExplainer(shapy_cat, tf.random.shuffle(x[i_train])[:num_background_samples].numpy())
                mean_dict = dict(mean=e.expected_value[0].numpy(), std=e.expected_value[1].numpy())
                mean = pd.concat([mean, pd.DataFrame(mean_dict, shap_index)])
                mean.to_pickle(mean_file)
                if not shap_index.isin(shap.index):
                    shap_values = e.shap_values(x[i_valid].numpy())
                    shap_dict = dict(sequence=sequence[i_valid].numpy().tolist(),
                                     mean=shap_values[0].sum(-1).tolist(),
                                     std=shap_values[1].sum(-1).tolist())
                    shap = pd.concat([shap, pd.DataFrame(shap_dict, shap_index.repeat(x[i_valid].shape[0]))])
                    shap.to_pickle(shap_file)

            # clear out memory and enable GPU determinism incase clearing undoes this setting
            tf.keras.backend.clear_session()
            tf.config.experimental.enable_op_determinism()

    # save performance measures and SHAP values
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
