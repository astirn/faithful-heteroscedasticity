import numpy as np
import tensorflow as tf


class RegressionCallback(tf.keras.callbacks.Callback):

    def __init__(self, same_line=False, early_stop_patience=0):
        """
        Custom performance monitoring callback with early stopping
        :param same_line: whether to print on the same line
        :param early_stop_patience: early stopping patience (a zero or negative value disables early stopping)
        """
        super().__init__()

        # save printing configuration
        self.same_line = same_line

        # early stopping configuration
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

        # update string
        update_string = 'Train RMSE = {:.4f} | '.format(logs['root_mean_squared_error'])
        update_string += 'Validation RMSE = {:.4f} | '.format(logs['val_root_mean_squared_error'])

        # early stopping logic
        if self.patience > 0:
            if tf.less(logs['val_root_mean_squared_error'], self.best):
                self.best = logs['val_root_mean_squared_error']
                self.wait = 0
                self.best_weights = self.model.get_weights()
            else:
                self.wait += 1
                if self.wait >= self.patience:
                    self.stopped_epoch = epoch
                    self.model.stop_training = True
            update_string += 'Best Validation RMSE = {:.4f} | '.format(self.best)
            update_string += 'Patience: {:d}/{:d}'.format(self.wait, self.patience)

        # test for NaN and Inf
        if tf.math.is_nan(logs['root_mean_squared_error']) or tf.math.is_inf(logs['root_mean_squared_error']):
            self.nan_inf = True
            self.stopped_epoch = epoch
            self.model.stop_training = True

        # print update
        if self.same_line:
            print('\r' + self.model.name + ' Epoch {:d} | '.format(epoch + 1) + update_string, end='')
        else:
            print(update_string)

    def on_train_end(self, logs=None):
        if self.same_line:
            print('')
        if self.nan_inf:
            print('Epoch {:d}: NaN or Inf detected!'.format(self.stopped_epoch + 1))
        else:
            print('Epoch {:d}: early stopping!'.format(self.stopped_epoch + 1))
        if self.stopped_epoch > 0:
            print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)
