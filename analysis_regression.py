import os
import numpy as np
import pandas as pd
import scipy.stats as stats


def get_target_dimension(df):
    return len([col for col in df.columns if col[:2] == 'y_'])


def steigers_test(y, y_hat_1, y_hat_2):
    r1 = stats.pearsonr(y, y_hat_1)[0]
    r2 = stats.pearsonr(y, y_hat_2)[0]
    r12 = stats.pearsonr(y_hat_1, y_hat_2)[0]

    # Steiger's test
    n = len(y)
    z1 = 0.5 * (np.log(1 + r1) - np.log(1 - r1))
    z2 = 0.5 * (np.log(1 + r2) - np.log(1 - r2))
    rm2 = (r1 ** 2 + r2 ** 2) / 2
    f = (1 - r12) / 2 / (1 - rm2)
    h = (1 - f * rm2) / (1 - rm2)
    z = abs(z1 - z2) * ((n - 3) / (2 * (1 - r12) * h)) ** 0.5
    log10_p = (stats.norm.logcdf(-z) + np.log(2)) / np.log10(np.e)

    return r1, r2, log10_p


def print_uci_table(df, file_name):

    # get column names
    if isinstance(df.index, pd.Index):
        columns = df.index.name
    elif isinstance(df.index, pd.MultiIndex):
        columns = df.index.names
    else:
        raise NotImplementedError

    # rearrange table for LaTeX
    df_latex = df.melt(id_vars=['Dataset'], value_vars=df.columns[1:], ignore_index=False)
    df_latex = df_latex.reset_index()
    df_latex = df_latex.pivot(index='Dataset', columns=columns + ['variable'], values='value')
    df_latex = pd.concat([df_latex[('Unit Variance Normal', 'MSE')],
                          df_latex[['Heteroscedastic Normal']],
                          df_latex[['Faithful Heteroscedastic Normal']]], axis=1)
    df_latex.style.hide(axis=1, names=True).to_latex(
        buf=os.path.join('tables', file_name),
        column_format='l' + ''.join(['|' + 'l' * len(df_latex[alg].columns) for alg in df_latex.columns.unique(0)]),
        hrules=True,
        # multicol_align='p{2cm}'
    )


def generate_uci_tables():

    # loop over datasets with predictions
    df_mse = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'regression')):
        measurements_file = os.path.join('experiments', 'regression', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            df_measurements = pd.read_pickle(measurements_file).sort_index()

            # drop index levels with just one unique value
            for level in df_measurements.index.names:
                if len(df_measurements.index.unique(level)) == 1:
                    df_measurements.set_index(df_measurements.index.droplevel(level), inplace=True)

            # null hypothesis values
            null_index = ['Unit Variance Normal']
            null_squared_errors = df_measurements.loc[null_index, 'squared_errors']
            # y_null, y_hat_null = get_targets_and_predictions(df_predictions, null_index)

            # loop over alternatives
            for index in df_measurements.index.unique():
                if isinstance(index, tuple):
                    index = pd.MultiIndex.from_tuples(tuples=[index], names=df_measurements.index.names)
                else:
                    index = pd.Index(data=[index], name=df_measurements.index.names)

                # MSE
                squared_errors = df_measurements.loc[index, 'squared_errors']
                mse = squared_errors.mean()
                p = stats.ttest_ind(squared_errors, null_squared_errors, equal_var=False, alternative='greater')[1]
                # p = stats.mannwhitneyu(squared_errors, null_squared_errors, alternative='greater')[1]
                mse = '\\sout{{{:.2g}}}'.format(mse) if p < 0.1 else '{:.2g}'.format(mse)
                p = '$H_0' if list(index) == null_index else '{:.2g}'.format(p)
                df_mse_add = pd.DataFrame({'Dataset': dataset, 'MSE': mse, 'Welch\'s $p$': p}, index)
                df_mse = pd.concat([df_mse, df_mse_add])

                # ECE

                # # Pearson
                # y, y_hat = get_targets_and_predictions(df_predictions, index)
                # assert np.min(np.abs(y - y_null)) == 0
                # _, r, log10_p = steigers_test(y, y_hat_null, y_hat)
                # df_pearson = pd.concat([df_pearson, pd.DataFrame({'Dataset': dataset,
                #                                                   'Pearson': r,
                #                                                   '$\\log_{10}(p)$': log10_p}, index)])

            # # ECE
            # ece = ExpectedCalibrationError(num_bins=6)

    print_uci_table(df_mse, file_name='regression_uci_mse.tex')
    # print_uci_table(df_pearson, file_name='regression_uci_pearson.tex')


if __name__ == '__main__':

    # output directory
    if not os.path.exists('tables'):
        os.mkdir('tables')

    # UCI tables
    generate_uci_tables()
