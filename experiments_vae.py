import argparse
import models
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa

from callbacks import RegressionCallback
from datasets import load_tensorflow_dataset
from metrics import RootMeanSquaredError

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl


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

    # load data
    x_clean_train, train_labels, x_clean_valid, valid_labels = load_tensorflow_dataset(args.dataset)

    # select a test set to plot sorted by class label
    i_test = tf.concat([tf.where(tf.equal(valid_labels, k))[0] for k in tf.unique(valid_labels)[0]], axis=0)
    i_test = tf.gather(i_test, tf.argsort(tf.gather(valid_labels, i_test)))

    # create heteroscedastic noise variance templates
    noise = (tf.ones_like(x_clean_train[0]) / 255).numpy()
    x, y = x_clean_train.shape[1] // 2, x_clean_train.shape[2] // 2
    noise[x - 2: x + 3, y: 2 * y + 1] = 0.25
    k = tf.unique(train_labels)[0].shape[0]
    noise_std = [tfa.image.rotate(noise, 2 * 3.14 / k * (y + 1), interpolation='bilinear') for y in range(k)]
    noise_std = tf.stack(noise_std, axis=0)

    # sample additive noise
    tf.keras.utils.set_random_seed(args.seed)
    x_corrupt_train = x_clean_train + tfd.Normal(loc=0, scale=tf.gather(noise_std, train_labels)).sample()
    x_corrupt_valid = x_clean_valid + tfd.Normal(loc=0, scale=tf.gather(noise_std, valid_labels)).sample()

    # initialize example image dictionary
    example_images = {
        'Data': {'clean': tf.gather(x_clean_valid, i_test), 'corrupt': tf.gather(x_corrupt_valid, i_test)},
        'Noise variance': {'clean': tf.zeros_like(noise_std), 'corrupt': noise_std ** 2},
        'Mean': {'clean': dict(), 'corrupt': dict()},
        'Std. deviation':  {'clean': dict(), 'corrupt': dict()},
    }

    # initialize/load optimization history
    opti_history_file = os.path.join(exp_path, 'optimization_history.pkl')
    optimization_history = pd.read_pickle(opti_history_file) if os.path.exists(opti_history_file) else pd.DataFrame()

    # loop over observation types
    measurements = pd.DataFrame()
    for observations in ['clean', 'corrupt']:
        print('******************** Observing: {:s} data ********************'.format(observations))

        # select training/validation data according to observation type
        if observations == 'clean':
            x_train, x_valid = x_clean_train, x_clean_valid
        elif observations == 'corrupt':
            x_train, x_valid = x_corrupt_train, x_corrupt_valid
        else:
            raise NotImplementedError

        # loop over models
        for mdl in [models.UnitVariance, models.Heteroscedastic, models.FaithfulHeteroscedastic]:

            # initialize models
            tf.keras.utils.set_random_seed(args.seed)
            model = mdl(dim_x=x_train.shape[1:], dim_y=x_train.shape[1:], dim_z=args.dim_z,
                        f_param=f_decoder, f_trunk=f_encoder)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          run_eagerly=args.debug, metrics=[RootMeanSquaredError()])

            # index for this model and observation type
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = pd.MultiIndex.from_tuples([(model_name, observations)], names=['Model', 'Observations'])

            # determine where to save model
            save_path = os.path.join(exp_path, observations, model.name)

            # if we are set to resume and a trained model and its optimization history exist, load the existing model
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
                optimization_history.drop(index, inplace=True)
                optimization_history = pd.concat([optimization_history, pd.DataFrame(
                    data={'Epoch': np.array(hist.epoch), 'RMSE': hist.history['RMSE']},
                    index=index.repeat(len(hist.epoch)))])

            # save local performance measurements
            tf.keras.utils.set_random_seed(args.seed)
            params = model.predict(x=x_valid, verbose=0)
            num_pixels = tf.cast(tf.reduce_prod(tf.shape(x_valid)[1:]), tf.float32)
            squared_errors = tf.einsum('abcd->a', (x_valid - params['mean']) ** 2) / num_pixels
            cdf_y = tf.einsum('abcd->a', model.predictive_distribution(**params).cdf(x_valid)) / num_pixels
            measurements = pd.concat([measurements, pd.DataFrame({'squared errors': squared_errors, 'F(y)': cdf_y},
                                                                 index.repeat(len(squared_errors)))])

            # save example images of the model's output
            tf.keras.utils.set_random_seed(args.seed)
            py_x = model.predictive_distribution(x=tf.gather(x_valid, i_test))
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            example_images['Mean'][observations].update({model_name: py_x.mean()})
            example_images['Std. deviation'][observations].update({model_name: py_x.stddev()})

    # save metrics, performance measures, and example images
    optimization_history.to_pickle(opti_history_file)
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
    with open(os.path.join(exp_path, 'example_images.pkl'), 'wb') as f:
        pickle.dump(example_images, f)
