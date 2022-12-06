import argparse
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from callbacks import RegressionCallback
from datasets import load_tensorflow_dataset
from metrics import RootMeanSquaredError
from models import get_models_and_configurations
from utils import model_config_dir, model_config_index, pretty_model_name

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl


def f_trunk(d_in, **kwargs):
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(kwargs['dim_z']), scale=1), reinterpreted_batch_ndims=1)
    # weight = 0.5 is a workaround for: https://github.com/tensorflow/probability/issues/1215
    dkl = tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=0.5)
    return tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='elu'),
        tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='elu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(kwargs['dim_z']), activation=None),
        tfpl.IndependentNormal(kwargs['dim_z'], activity_regularizer=dkl),
    ])


def f_param(d_in, d_out, f_out, **kwargs):
    m = tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(1568, activation='elu'),
        tf.keras.layers.Reshape([7, 7, 32]),
        tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding='same', activation='elu'),
        tf.keras.layers.Conv2DTranspose(1, 5, strides=2, padding='same', activation=f_out),
    ])
    assert m.output_shape[1:] == d_out
    return m


def f_vae(d_in, d_out, f_out, **kwargs):
    vae = f_trunk(d_in, **kwargs)
    vae.add(f_param(vae.output_shape[1:], d_out, f_out, **kwargs))
    return vae


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dataset', type=str, default='mnist', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--dim_z', type=int, default=16, help='number of latent dimensions')
    parser.add_argument('--epochs', type=int, default=2000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--replace', action='store_true', default=False, help='whether to replace saved model')
    parser.add_argument('--seed', type=int, default=123456789, help='random number seed for reproducibility')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'vae', args.dataset)
    os.makedirs(exp_path, exist_ok=True)

    # enable GPU determinism
    tf.config.experimental.enable_op_determinism()

    # models and configurations to run
    nn_kwargs = dict(dim_z=args.dim_z, f_trunk=f_trunk, f_param=f_param)
    models_and_configurations = get_models_and_configurations(nn_kwargs)
    # nn_kwargs = dict(dim_z=args.dim_z, f_trunk=None, f_param=f_vae)
    # models_and_configurations += get_models_and_configurations(nn_kwargs)

    # load data
    x_clean_train, train_labels, x_clean_valid, valid_labels = load_tensorflow_dataset(args.dataset)
    dim_x = x_clean_train.shape[1:]

    # create heteroscedastic noise variance templates
    noise = (tf.ones_like(x_clean_train[0]) / 255).numpy()
    x, y = dim_x[0] // 2, dim_x[1] // 2
    noise[x - 2: x + 3, y: 2 * y + 1] = 0.25 * (tf.reduce_max(x_clean_train) - tf.reduce_min(x_clean_train))
    k = tf.unique(train_labels)[0].shape[0]
    noise_std = [tfa.image.rotate(noise, 2 * 3.14 / k * (y + 1), interpolation='bilinear') for y in range(k)]
    noise_std = tf.stack(noise_std, axis=0)

    # sample additive noise
    tf.keras.utils.set_random_seed(args.seed)
    x_corrupt_train = x_clean_train + tfd.Normal(loc=0, scale=tf.gather(noise_std, train_labels)).sample()
    x_corrupt_valid = x_clean_valid + tfd.Normal(loc=0, scale=tf.gather(noise_std, valid_labels)).sample()

    # initialize performance containers
    measurements_df = pd.DataFrame()
    measurements_dict = {
        'Class labels': valid_labels.numpy(),
        'Data': {'clean': x_clean_valid.numpy(), 'corrupt': x_corrupt_valid.numpy()},
        'Noise variance': {'clean': tf.zeros_like(noise_std).numpy(), 'corrupt': noise_std.numpy() ** 2},
        'Mean': dict(),
        'Std.': dict(),
        'Z':  dict(),
    }

    # initialize/load optimization history
    opti_history_file = os.path.join(exp_path, 'optimization_history.pkl')
    optimization_history = pd.read_pickle(opti_history_file) if os.path.exists(opti_history_file) else pd.DataFrame()

    # loop over observation types
    for observations in ['clean', 'corrupt']:

        # select training/validation data according to observation type
        if observations == 'clean':
            x_train, x_valid = x_clean_train, x_clean_valid
        elif observations == 'corrupt':
            x_train, x_valid = x_corrupt_train, x_corrupt_valid
        else:
            raise NotImplementedError

        # loop over models/architectures/configurations
        for mag in models_and_configurations:

            # model configuration (seed and GPU determinism ensures architectures are identically initialized)
            tf.keras.utils.set_random_seed(args.seed)
            model = mag['model'](dim_x=dim_x, dim_y=dim_x, **mag['model_kwargs'], **mag['nn_kwargs'])
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate), metrics=[RootMeanSquaredError()])

            # index for this model and configuration
            model_name = pretty_model_name(model, mag['model_kwargs'])
            index, index_str = model_config_index(model_name, model.model_class,
                                                  **{**{'Observations': observations}, **mag['nn_kwargs']})
            print('***** {:s} *****'.format(index_str))

            # determine where to save model
            save_path = os.path.join(exp_path, observations)
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
                tf.keras.utils.set_random_seed(args.seed)
                hist = model.fit(x=x_train, y=x_train, validation_data=(x_valid, x_valid),
                                 batch_size=args.batch_size, epochs=args.epochs, verbose=0,
                                 callbacks=[RegressionCallback(early_stop_patience=100)])
                model.save_weights(os.path.join(save_path, 'best_checkpoint'))
                if index.isin(optimization_history.index):
                    optimization_history.drop(index, inplace=True)
                optimization_history = pd.concat([optimization_history, pd.DataFrame(
                    data={'Epoch': np.array(hist.epoch), 'RMSE': hist.history['RMSE']},
                    index=index.repeat(len(hist.epoch)))])
                optimization_history.to_pickle(opti_history_file)

            # generate predictions on the validation set
            tf.keras.utils.set_random_seed(args.seed)
            params = model.predict(x=x_valid, verbose=0)

            # update measurements DataFrame (high dimensional objects should go in the dictionary)
            num_pixels = tf.cast(tf.reduce_prod(tf.shape(x_valid)[1:]), tf.float32)
            py_x = tfd.Independent(model.predictive_distribution(**params), tf.rank(x_valid) - 1)
            measurements_df = pd.concat([measurements_df, pd.DataFrame({
                'log p(y|x)': py_x.log_prob(x_valid),
                'squared errors': tf.einsum('abcd->a', (x_valid - py_x.mean()) ** 2) / num_pixels,
            }, index.repeat(py_x.batch_shape))])

            # update measurements dictionary
            measurements_dict['Mean'].update({index_str: py_x.mean().numpy()})
            measurements_dict['Std.'].update({index_str: py_x.stddev().numpy()})
            z_scores = (x_valid - py_x.mean()) / py_x.stddev()
            measurements_dict['Z'].update({index_str: z_scores.numpy()})

    # save performance measures and model outputs
    measurements_df.to_pickle(os.path.join(exp_path, 'measurements_df.pkl'))
    with open(os.path.join(exp_path, 'measurements_dict.pkl'), 'wb') as f:
        pickle.dump(measurements_dict, f)
