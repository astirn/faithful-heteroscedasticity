import numpy as np
import tensorflow as tf


class RegressionCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_freq=0, use_train_as_valid=False, early_stop_patience=0):
        """
        Custom performance monitoring callback with early stopping
        :param validation_freq: at which epoch frequency to print and do model checking (must align with keras fit)
        :param use_train_as_valid: whether to treat training data as validation data
        :param early_stop_patience: early stopping patience (a zero or negative value disables early stopping)
        """
        super().__init__()

        # configure and initialize
        self.validation_freq = validation_freq
        self.use_train_as_valid = use_train_as_valid
        self.patience = early_stop_patience
        self.nan_inf = False
        self.wait = 0
        self.stopped_epoch = 0
        self.best_weights = None
        self.best = None

    def on_train_begin(self, logs=None):
        # reinitialization code that allows instance to be reused
        self.nan_inf = False
        self.wait = 0
        self.stopped_epoch = 0
        self.best = np.inf
        self.best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        epoch += 1
        if epoch % self.validation_freq == 0:

            # get training and validation performance
            train_rmse = logs['root_mean_squared_error']
            valid_rmse = train_rmse if self.use_train_as_valid else logs['val_root_mean_squared_error']

            # update string
            update_string = 'Train RMSE = {:.4f} | '.format(train_rmse)
            update_string += 'Validation RMSE = {:.4f}'.format(valid_rmse)

            # early stopping logic
            if self.patience > 0:
                if tf.less(valid_rmse, self.best):
                    self.best = valid_rmse
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                update_string += ' | Best Validation RMSE = {:.4f}'.format(self.best)
                update_string += ' | Patience: {:d}/{:d}'.format(self.wait, self.patience)

            # test for NaN and Inf
            if tf.math.is_nan(train_rmse) or tf.math.is_inf(train_rmse):
                self.nan_inf = True
                self.stopped_epoch = epoch
                self.model.stop_training = True

            # print update
            print('\r' + self.model.name + ' Epoch {:d} | '.format(epoch) + update_string, end='')

    def on_train_end(self, logs=None):
        if self.nan_inf:
            print('\nEpoch {:d}: NaN or Inf detected!'.format(self.stopped_epoch + 1))
        else:
            print('\nEpoch {:d}: early stopping!'.format(self.stopped_epoch + 1))
        if self.stopped_epoch > 1:
            print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)
