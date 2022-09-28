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

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl
from utils import pretty_model_name


def f_encoder(d_in, dim_z, **kwargs):
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(dim_z), scale=1), reinterpreted_batch_ndims=1)
    # weight = 0.5 is a workaround for: https://github.com/tensorflow/probability/issues/1215
    dkl = tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=0.5)
    return tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Conv2D(32, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, 7, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(dim_z), activation=None),
        tfpl.IndependentNormal(dim_z, activity_regularizer=dkl)
    ])


def f_decoder(d_in, d_out, f_out, **kwargs):
    return tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Reshape([1, 1, d_in]),
        tf.keras.layers.Conv2DTranspose(64, 7, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Conv2DTranspose(64, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(64, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(32, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(32, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2DTranspose(32, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(1, 5, strides=1, padding='same', activation=f_out),
    ])


def f_encoder_decoder(d_in, d_out, f_out, **kwargs):
    m = f_encoder(d_in, **kwargs)
    m.add(f_decoder(kwargs.pop('dim_z'), d_out, f_out, **kwargs))
    return m


if __name__ == '__main__':

    # script arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
    parser.add_argument('--dataset', type=str, default='mnist', help='which dataset to use')
    parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
    parser.add_argument('--dim_z', type=int, default=10, help='number of latent dimensions')
    parser.add_argument('--epochs', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=5e-5, help='learning rate')
    parser.add_argument('--replace', action='store_true', default=False, help='whether to replace saved model')
    parser.add_argument('--seed', type=int, default=112358, help='random number seed for reproducibility')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'vae', args.dataset, str(args.dim_z))
    os.makedirs(exp_path, exist_ok=True)

    # enable GPU determinism
    tf.config.experimental.enable_op_determinism()

    # models and configurations to run
    models_and_configurations = get_models_and_configurations(dict(d_hidden=(128, 32)))

    # load data
    x_clean_train, train_labels, x_clean_valid, valid_labels = load_tensorflow_dataset(args.dataset)
    dim_x = x_clean_train.shape[1:]

    # create heteroscedastic noise variance templates
    noise = (tf.ones_like(x_clean_train[0]) / 255).numpy()
    x, y = dim_x[0] // 2, dim_x[1] // 2
    noise[x - 2: x + 3, y: 2 * y + 1] = 0.25
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
        'Class labels': valid_labels,
        'Data': {'clean': x_clean_valid, 'corrupt': x_corrupt_valid},
        'Noise variance': {'clean': tf.zeros_like(noise_std), 'corrupt': noise_std ** 2},
        'Mean': {'clean': dict(), 'corrupt': dict()},
        'Std.':  {'clean': dict(), 'corrupt': dict()},
        'Z':  {'clean': dict(), 'corrupt': dict()},
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
            print('***** Observing {:s} data w/ a {:s} network *****'.format(observations, mag['architecture']))

            # model configuration (seed and GPU determinism ensures architectures are identically initialized)
            tf.keras.utils.set_random_seed(args.seed)
            if mag['architecture'] == 'separate':
                model = mag['model'](dim_x=dim_x, dim_y=dim_x, f_trunk=None, f_param=f_encoder_decoder,
                                     dim_z=args.dim_z, **mag['config'], **mag['nn_kwargs'])
            elif mag['architecture'] in {'single', 'shared'}:
                model = mag['model'](dim_x=dim_x, dim_y=dim_x, f_trunk=f_encoder, f_param=f_decoder,
                                     dim_z=args.dim_z, **mag['config'], **mag['nn_kwargs'])
            else:
                raise NotImplementedError
            optimizer = tf.keras.optimizers.Adam(args.learning_rate)
            model.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[RootMeanSquaredError()])

            # index for this model and observation type
            model_name = pretty_model_name(model)
            index = pd.MultiIndex.from_tuples([(model_name, mag['architecture'], observations)],
                                              names=['Model', 'Architecture', 'Observations'])

            # determine where to save model
            save_path = os.path.join(exp_path, observations, ''.join([model.name, mag['architecture'].capitalize()]))

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
                'squared errors': tf.einsum('abcd->a', (x_valid - params['mean']) ** 2) / num_pixels,
            }, index.repeat(py_x.batch_shape))])

            # update measurements dictionary
            measurements_dict['Mean'][observations].update({model_name + ' ' + mag['architecture']: params['mean']})
            measurements_dict['Std.'][observations].update({model_name + ' ' + mag['architecture']: params['std']})
            z_scores = (x_valid - params['mean']) / params['std']
            measurements_dict['Z'][observations].update({model_name + ' ' + mag['architecture']: z_scores})

        # save performance measures and model outputs
        measurements_df.to_pickle(os.path.join(exp_path, 'measurements_df.pkl'))
        with open(os.path.join(exp_path, 'measurements_dict.pkl'), 'wb') as f:
            pickle.dump(measurements_dict, f)
