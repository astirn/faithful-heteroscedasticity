import argparse
import os
import pickle
import zlib

import pandas as pd
import tensorflow as tf

from callbacks import RegressionCallback
from data_vae import load_data
from metrics import MeanLogLikelihood, RootMeanSquaredError, ExpectedCalibrationError
from models_vae import encoder_dense, decoder_dense, NormalVAE, StudentVAE, GammaNormalVAE

# script arguments
parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=2048, help='batch size')
parser.add_argument('--dataset', type=str, default='mnist', help='which dataset to use')
parser.add_argument('--debug', action='store_true', default=False, help='run eagerly')
parser.add_argument('--dim_z', type=int, default=10, help='number of latent dimensions')
parser.add_argument('--num_mc_samples', type=int, default=20, help='number of MC samples')
parser.add_argument('--num_trials', type=int, default=5, help='number of trials')
parser.add_argument('--seed', type=int, default=853211, help='random seed')
parser.add_argument('--replace', action='store_true', default=False, help='whether to replace existing results')
args = parser.parse_args()

# make experimental directory base path
exp_path = os.path.join('experiments', 'vae', args.dataset)
os.makedirs(exp_path, exist_ok=True)

# models and configurations to run
nn_kwargs = {'topology': 'dense', 'encoder_arch': [512, 256, 128], 'decoder_arch': [128, 256, 512]}
models_and_configs = [
    {'model': NormalVAE, 'config': {'optimization': 'first-order'}},
    {'model': NormalVAE, 'config': {'optimization': 'second-order-mean'}},
    {'model': StudentVAE, 'config': {'min_df': 3}},
    {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': False, 'prior_scale': 1e-3}},
    # {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 0.25}},
    # {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 0.50}},
    {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 0.75}},
    # {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 0.85}},
    {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 1.00}},
    # {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 1.15}},
    {'model': GammaNormalVAE, 'config': {'min_df': 3, 'empirical_bayes': True, 'prior_scale': 1.25}},
]

# load data folds
ds_train, ds_valid = load_data(args.dataset, batch_size=args.batch_size)

# loop over trials
results = pd.DataFrame()
for trial in range(1, args.num_trials + 1):
    trial_path = os.path.join(exp_path, 'trial_' + str(trial))

    # loop over the models and configurations
    for model_and_config in models_and_configs:
        print('\n********* Trial {:d} *********'.format(trial))

        # each trial within a fold gets the same random seed
        tf.keras.utils.set_random_seed(trial * args.seed)

        # configure and build model
        model = model_and_config['model'](
            dim_x=(28, 28, 1),
            dim_z=args.dim_z,
            encoder=encoder_dense if nn_kwargs['topology'] == 'dense' else None,
            encoder_arch=nn_kwargs['encoder_arch'],
            decoder=decoder_dense if nn_kwargs['topology'] == 'dense' else None,
            decoder_arch=nn_kwargs['decoder_arch'],
            num_mc_samples=20,
            **model_and_config['config']
        )
        optimizer = tf.keras.optimizers.Adam(5e-5)
        model.compile(optimizer=optimizer, run_eagerly=args.debug, metrics=[
            MeanLogLikelihood(),
            RootMeanSquaredError(),
            ExpectedCalibrationError(),
        ])

        # determine directory where to save model
        nn_dir = zlib.crc32(str(nn_kwargs).encode('utf'))
        model_dir = os.path.join(trial_path, str(nn_dir), model.name)

        # if we are set to resume and the model directory already contains a saved model, load it
        if not bool(args.replace) and os.path.exists(os.path.join(model_dir, 'checkpoint')):
            print(model.name + ' exists. Loading...')
            model.load_weights(os.path.join(model_dir, 'best_checkpoint'))
            with open(os.path.join(model_dir, 'hist.pkl'), 'rb') as f:
                history = pickle.load(f)

        # otherwise, train and save the model
        else:
            validation_freq = 1
            hist = model.fit(x=ds_train, validation_data=ds_valid, validation_freq=validation_freq, epochs=int(20e3),
                             verbose=0,
                             callbacks=[RegressionCallback(validation_freq=validation_freq, early_stop_patience=20)])
            model.save_weights(os.path.join(model_dir, 'best_checkpoint'))
            history = hist.history
            with open(os.path.join(model_dir, 'hist.pkl'), 'wb') as f:
                pickle.dump(history, f)

        # test model
        test_metrics = model.evaluate(x=ds_valid, verbose=0)
        print('Test LL = {:.4f} | Test RMSE = {:.4f} | Test ECE = {:.4f}'.format(*test_metrics))

        # update results table
        new_results = pd.DataFrame(
            data={'Model': model.name, 'Architecture': str(nn_kwargs), 'Epochs': len(history['LL']),
                  'LL': test_metrics[0], 'RMSE': test_metrics[1], 'ECE': test_metrics[2]},
            index=pd.MultiIndex.from_arrays([[trial]], names=['trial']))
        results = pd.concat([results, new_results])

# save the results
results.to_pickle(os.path.join(exp_path, 'results.pkl'))
