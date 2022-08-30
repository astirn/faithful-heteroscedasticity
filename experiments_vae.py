import argparse
import models
import os
import pickle

import tensorflow as tf
import tensorflow_addons as tfa

from callbacks import RegressionCallback
from datasets import load_tensorflow_dataset
from metrics import RootMeanSquaredError, ExpectedCalibrationError

from tensorflow_probability import distributions as tfd
from tensorflow_probability import layers as tfpl


# encoder network
def f_encoder(d_in, dim_z, **kwargs):
    prior = tfd.Independent(tfd.Normal(loc=tf.zeros(dim_z), scale=1), reinterpreted_batch_ndims=1)
    return tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Conv2D(32, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(32, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 5, strides=1, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(64, 5, strides=2, padding='same', activation='relu'),
        tf.keras.layers.Conv2D(128, 7, strides=1, padding='valid', activation='relu'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(dim_z), activation=None),
        # workaround for: https://github.com/tensorflow/probability/issues/1215
        tfpl.IndependentNormal(dim_z,
                               activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=0.5))
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
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--seed_data', type=int, default=112358, help='seed to generate folds')
    parser.add_argument('--seed_init', type=int, default=853211, help='seed to initialize model')
    args = parser.parse_args()

    # make experimental directory base path
    exp_path = os.path.join('experiments', 'vae', args.dataset, str(args.dim_z))
    os.makedirs(exp_path, exist_ok=True)

    # load data
    x_clean_train, train_labels, x_clean_valid, valid_labels = load_tensorflow_dataset(args.dataset)

    # select a test set to plot sorted by class label
    i_test = tf.concat([tf.where(tf.equal(valid_labels, k))[0] for k in tf.unique(valid_labels)[0]], axis=0)
    i_test = tf.gather(i_test, tf.argsort(tf.gather(valid_labels, i_test)))

    # create heteroscedastic noise variance templates
    noise = tf.tile(tf.expand_dims(tf.linspace(1 / 255, 0.25, 28), axis=1), [1, tf.shape(x_clean_train)[1]])
    num_classes = tf.unique(train_labels)[0].shape[0]
    noise_std = [tfa.image.rotate(noise, 360 / num_classes * y, fill_mode='nearest') for y in range(num_classes)]
    noise_std = tf.expand_dims(tf.stack(noise_std, axis=0), axis=-1)

    # sample additive noise
    tf.keras.utils.set_random_seed(args.seed_data)
    x_corrupt_train = x_clean_train + tfd.Normal(loc=0, scale=tf.gather(noise_std, train_labels)).sample()
    x_corrupt_valid = x_clean_valid + tfd.Normal(loc=0, scale=tf.gather(noise_std, valid_labels)).sample()

    # initialize test measurements
    measurements = {
        'Data': {'clean': tf.gather(x_clean_valid, i_test), 'corrupt': tf.gather(x_corrupt_valid, i_test)},
        'Noise variance': {'clean': tf.zeros_like(noise_std), 'corrupt': noise_std ** 2},
        'Mean': {'clean': dict(), 'corrupt': dict()},
        'Std. deviation':  {'clean': dict(), 'corrupt': dict()},
    }

    # loop over observation types
    for observation in ['clean', 'corrupt']:
        print('******************** Observing: {:s} data ********************'.format(observation))

        # select training/validation data according to observation type
        if observation == 'clean':
            x_train, x_valid = x_clean_train, x_clean_valid
        elif observation == 'corrupt':
            x_train, x_valid = x_corrupt_train, x_corrupt_valid
        else:
            raise NotImplementedError

        # loop over models
        for mdl in [models.UnitVariance, models.Heteroscedastic, models.FaithfulHeteroscedastic]:

            # initialize model
            tf.keras.utils.set_random_seed(args.seed_init)
            model = mdl(dim_x=x_train.shape[1:], dim_y=x_train.shape[1:], dim_z=args.dim_z,
                        f_param=f_decoder, f_trunk=f_encoder)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                          run_eagerly=args.debug,
                          metrics=[RootMeanSquaredError(), ExpectedCalibrationError()])

            # train the model
            model.fit(x=x_train, y=x_train, validation_data=(x_valid, x_valid),
                      batch_size=args.batch_size, epochs=args.epochs, verbose=0,
                      callbacks=[RegressionCallback(early_stop_patience=100)])

            # measure performance
            py_x = model.predictive_distribution(x=tf.gather(x_valid, i_test))
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            measurements['Mean'][observation].update({model_name: py_x.mean()})
            measurements['Std. deviation'][observation].update({model_name: py_x.stddev()})

    # save performance measures
    with open(os.path.join(exp_path, 'measurements.pkl'), 'wb') as f:
        pickle.dump(measurements, f)
