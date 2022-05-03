import os
import glob
import pickle
import zipfile
import warnings
import numpy as np
import pandas as pd
from urllib import request


def trim_eol_whitespace(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(' \n', '\n') for line in lines]
    with open(data_file, 'w') as f:
        f.writelines(lines)


def decimal_comma_to_decimal_point(data_file):
    with open(data_file, 'r') as f:
        lines = f.readlines()
    lines = [line.replace(',', '.') for line in lines]
    with open(data_file, 'w') as f:
        f.writelines(lines)


REGRESSION_DATA = {
    'boston':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data',
         'dir_after_unzip': None,
         'data_file': 'housing.data',
         'parse_args': {'sep': ' ', 'header': None, 'skipinitialspace': True},
         'target_cols': [-1]},
    'carbon':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00448/carbon_nanotubes.csv',
         'dir_after_unzip': None,
         'data_file': 'carbon_nanotubes.csv',
         'formatter': decimal_comma_to_decimal_point,
         'parse_args': {'sep': ';'},
         'target_cols': [-1, -2, -3]},
    'concrete':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls',
         'dir_after_unzip': None,
         'data_file': 'Concrete_Data.xls',
         'parse_args': dict(),
         'target_cols': [-1]},
    'energy':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx',
         'dir_after_unzip': None,
         'data_file': 'ENB2012_data.xlsx',
         'parse_args': dict(),
         'target_cols': [-1, -2]},
    'naval':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00316/UCI%20CBM%20Dataset.zip',
         'dir_after_unzip': 'UCI CBM Dataset',
         'data_file': 'data.txt',
         'parse_args': {'sep': ' ', 'header': None, 'skipinitialspace': True},
         'target_cols': [-1, -2]},
    'power plant':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00294/CCPP.zip',
         'dir_after_unzip': 'CCPP',
         'data_file': 'Folds5x2_pp.xlsx',
         'parse_args': dict(),
         'target_cols': [-1]},
    'protein':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00265/CASP.csv',
         'dir_after_unzip': None,
         'data_file': 'CASP.csv',
         'parse_args': dict(),
         'target_cols': [1]},
    'superconductivity':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00464/superconduct.zip',
         'dir_after_unzip': None,
         'data_file': 'train.csv',
         'parse_args': dict(),
         'target_cols': [-1]},
    'wine-red':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',
         'dir_after_unzip': None,
         'data_file': 'winequality-red.csv',
         'parse_args': {'sep': ';'},
         'target_cols': [-1]},
    'wine-white':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-white.csv',
         'dir_after_unzip': None,
         'data_file': 'winequality-white.csv',
         'parse_args': {'sep': ';'},
         'target_cols': [-1]},
    'yacht':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00243/yacht_hydrodynamics.data',
         'dir_after_unzip': None,
         'data_file': 'yacht_hydrodynamics.data',
         'formatter': trim_eol_whitespace,
         'parse_args': {'sep': ' ', 'header': None, 'skipinitialspace': True},
         'target_cols': [-1]},
    'year':
        {'url': 'https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip',
         'dir_after_unzip': None,
         'data_file': 'YearPredictionMSD.txt',
         'parse_args': dict(),
         'target_cols': [1]},
}


def download_all(force_download=False):

    # make data directory if it doesn't yet exist
    if not os.path.exists('data'):
        os.mkdir('data')

    # download all regression data experiments
    for key in REGRESSION_DATA.keys():
        data_dir = os.path.join('data', key)
        if not os.path.exists(data_dir):
            os.mkdir(data_dir)
        file = os.path.join(data_dir, REGRESSION_DATA[key]['url'].split('/')[-1])
        if os.path.exists(file) and force_download:
            os.remove(file)
        elif os.path.exists(file) and not force_download:
            print(file.split(os.sep)[-1], 'already exists.')
            continue
        print('Downloading', file.split(os.sep)[-1])
        request.urlretrieve(REGRESSION_DATA[key]['url'], file)
    print('Downloads complete!')


def load_data(data_dir, dir_after_unzip, data_file, parse_args, **kwargs):

    # save the base data directory as the save directory, since data_dir might be modified below
    save_dir = data_dir

    # find any zip files
    zip_files = glob.glob(os.path.join(data_dir, '*.zip'))
    assert len(zip_files) <= 1

    # do we need to unzip?
    if len(zip_files) or dir_after_unzip is not None:

        # unzip it
        with zipfile.ZipFile(zip_files[0], 'r') as f:
            f.extractall(data_dir)

        # update data directory if required
        if dir_after_unzip is not None:
            data_dir = os.path.join(data_dir, dir_after_unzip)

    # correct formatting issues if necessary
    if 'formatter' in kwargs.keys() and kwargs['formatter'] is not None:
        kwargs['formatter'](os.path.join(data_dir, data_file))

    # process files according to type
    if os.path.splitext(data_file)[-1] in {'.csv', '.data', '.txt'}:
        df = pd.read_csv(os.path.join(data_dir, data_file), **parse_args)
    elif os.path.splitext(data_file)[-1] in {'.xls', '.xlsx'}:
        df = pd.read_excel(os.path.join(data_dir, data_file))
    else:
        warnings.warn('Type Not Supported: ' + data_file)
        return

    # convert to numpy arrays
    xy = df.to_numpy(dtype=np.float32)
    y = xy[:, kwargs['target_cols']]
    x_indices = list(range(xy.shape[1]))
    for i in kwargs['target_cols']:
        x_indices.pop(i)
    x = xy[:, x_indices]

    # save data
    with open(os.path.join(save_dir, save_dir.split(os.sep)[-1] + '.pkl'), 'wb') as f:
        pickle.dump({'data': x, 'target': y}, f)


def generate_toy_data(num_samples=500, sparse=False):
    def data_mean(x):
        return x * np.sin(x)

    def data_std(x):
        return 0.1 + np.abs(0.5 * x)

    # sample training data
    if not sparse:
        x_data = np.random.uniform(0, 10, size=num_samples)
    else:
        x_data = np.random.uniform(2.5, 7.5, size=num_samples - 2)
    noise = np.random.normal(scale=data_std(x_data))
    y_data = data_mean(x_data) + noise
    if sparse:
        x_isolated = np.array([0.5, 9.5])
        x_data = np.concatenate([x_data, x_isolated], axis=0)
        y_data = np.concatenate([y_data, data_mean(x_isolated)], axis=0)

    # generate evaluation points with the associated actual mean and standard deviation
    x_eval = np.linspace(-4, 14, 250)
    true_mean = data_mean(x_eval)
    true_std = data_std(x_eval)

    # process return tuple
    return_tuple = (x_data, y_data, x_eval, true_mean, true_std)
    return_tuple = (np.expand_dims(np.float32(x), axis=-1) for x in return_tuple)

    return return_tuple


if __name__ == '__main__':

    # download all the data
    download_all()

    # process all the data
    for key in REGRESSION_DATA.keys():
        load_data(data_dir=os.path.join('data', key), **REGRESSION_DATA[key])
    print('Processing complete!')
