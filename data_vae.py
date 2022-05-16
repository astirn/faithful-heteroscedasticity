import tensorflow as tf
import tensorflow_datasets as tfds


def load_data(dataset):
    ds_train = tfds.load(name=dataset, split=tfds.Split.TRAIN, data_dir='data')
    ds_valid = tfds.load(name=dataset, split=tfds.Split.TEST, data_dir='data')
    x_train = [tf.cast(ele['image'], tf.float32) for ele in ds_train.batch(len(ds_train))][0]
    x_valid = [tf.cast(ele['image'], tf.float32) for ele in ds_valid.batch(len(ds_valid))][0]
    x_train = x_train / tf.reduce_max(x_train)
    x_valid = x_valid / tf.reduce_max(x_train)
    ds_train = tf.data.Dataset.from_tensor_slices({'x': x_train}).batch(2048)
    ds_valid = tf.data.Dataset.from_tensor_slices({'x': x_valid}).batch(2048)

    return ds_train, ds_valid
