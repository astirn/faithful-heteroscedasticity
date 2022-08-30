import numpy as np
import tensorflow as tf


class RegressionCallback(tf.keras.callbacks.Callback):

    def __init__(self, validation_freq=1, early_stop_patience=0):
        super().__init__()

        # configure and initialize
        self.monitor = 'val_RMSE'
        self.validation_freq = validation_freq
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

            # update string
            update_string = self.model.name + ' | Epoch {:d}'.format(epoch)
            for key, val in logs.items():
                update_string += ' | {:s} = {:.4f}'.format(key, val)

            # early stopping logic
            if self.patience > 0:
                if tf.less(logs[self.monitor], self.best):
                    self.best = logs[self.monitor]
                    self.wait = 0
                    self.best_weights = self.model.get_weights()
                else:
                    self.wait += 1
                    if self.wait >= self.patience:
                        self.stopped_epoch = epoch
                        self.model.stop_training = True
                update_string += ' | Best {:s} = {:.4f}'.format(self.monitor, self.best)
                update_string += ' | Patience: {:d}/{:d}'.format(self.wait, self.patience)

            # test for NaN and Inf
            if tf.math.is_nan(logs['RMSE']) or tf.math.is_inf(logs['RMSE']):
                self.nan_inf = True
                self.stopped_epoch = epoch
                self.model.stop_training = True

            # print update
            print('\r' + update_string, end='')

    def on_train_end(self, logs=None):
        if self.nan_inf:
            print('\nEpoch {:d}: NaN or Inf detected!'.format(self.stopped_epoch + 1))
        else:
            print('\nFinished!')
        if self.stopped_epoch > self.validation_freq:
            print('Restoring model weights from the end of the best epoch.')
            self.model.set_weights(self.best_weights)
