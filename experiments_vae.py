import argparse
import models
import os

import pandas as pd
import tensorflow as tf

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
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(tfpl.IndependentNormal.params_size(dim_z), activation=None),
        # workaround for: https://github.com/tensorflow/probability/issues/1215
        tfpl.IndependentNormal(dim_z,
                               activity_regularizer=tfpl.KLDivergenceRegularizer(prior, use_exact_kl=True, weight=0.5))
    ])


def f_decoder(d_in, d_out, f_out, **kwargs):
    return tf.keras.Sequential(layers=[
        tf.keras.layers.InputLayer(d_in),
        tf.keras.layers.Dense(128, activation='elu'),
        tf.keras.layers.Dense(256, activation='elu'),
        tf.keras.layers.Dense(tf.reduce_prod(d_out), activation=f_out),
        tf.keras.layers.Reshape(d_out)
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
    exp_path = os.path.join('experiments', 'vae', args.dataset)
    os.makedirs(exp_path, exist_ok=True)

    # load data
    x_clean_train, x_clean_valid = load_tensorflow_dataset(args.dataset)

    # loop over observation types
    measurements = pd.DataFrame()
    for observation in ['clean', 'corrupted']:
        print('******************** Observing: {:s} data ********************'.format(observation))

        # select training/validation data according to observation type
        if observation == 'clean':
            x_train, x_valid = x_clean_train, x_clean_valid
        elif observation == 'corrupted':
            raise NotImplementedError
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

            # index for this model and observation type
            model_name = ''.join(' ' + char if char.isupper() else char.strip() for char in model.name).strip()
            index = pd.MultiIndex.from_tuples([(model_name, observation)], names=['Model', 'Observation'])

            # measure performance
            py_x = model.predictive_distribution(x=x_valid)
            measurements = pd.concat([measurements, pd.DataFrame(
                data={'x': x_valid.numpy().tolist(),
                      'Mean': py_x.mean().numpy().tolist(),
                      'Std. Deviation': py_x.stddev().numpy().tolist()},
                index=index.repeat(x_valid.shape[0]))])

    # save performance measures
    measurements.to_pickle(os.path.join(exp_path, 'measurements.pkl'))
