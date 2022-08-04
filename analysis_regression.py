import os
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, ttest_ind, mannwhitneyu


def get_target_dimension(df):
    return len([col for col in df.columns if col[:2] == 'y_'])


def get_squared_errors(df, idx):
    df = df.loc[idx]
    sq_errors = np.zeros(len(df))
    for i in range(get_target_dimension(df)):
        sq_errors += (df['y_' + str(i)] - df['mean_' + str(i)]) ** 2
    return sq_errors


def print_uci_table(df):

    test = df.melt(id_vars=['Dataset'], value_vars=['MSE', 'p-value'], ignore_index=False)
    test = test.reset_index()
    del test['Architecture']
    test = test.pivot(index='Dataset', columns=['Model', 'variable'], values='value')
    test = test[['UnitVarianceNormal', 'HeteroscedasticNormal', 'FaithfulHeteroscedasticNormal']]
    test.style.hide(axis=1, names=True).format("{:.3g}".format).to_latex(
        buf=os.path.join('tables', 'regression_uci_mse.tex'),
        hrules=True,
    )


def generate_uci_tables():

    # loop over datasets with predictions
    df_mse = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'regression')):
        predictions_file = os.path.join('experiments', 'regression', dataset, 'predictions.pkl')
        if os.path.exists(predictions_file):
            df_predictions = pd.read_pickle(predictions_file).sort_index()

            # MSE
            null_index = pd.MultiIndex.from_tuples(
                tuples=[('UnitVarianceNormal', "{'n_hidden': 2, 'd_hidden': 50, 'f_hidden': 'elu'}")],
                names=df_predictions.index.names)
            null_squared_errors = get_squared_errors(df_predictions, null_index)
            for index in df_predictions.index.unique():
                index = pd.MultiIndex.from_tuples(tuples=[index], names=df_predictions.index.names)
                squared_errors = get_squared_errors(df_predictions, index)
                # p_value = ttest_ind(squared_errors, null_squared_errors, equal_var=False, alternative='greater')[1]
                p_value = mannwhitneyu(squared_errors, null_squared_errors, alternative='greater')[1]
                df_mse = pd.concat([df_mse, pd.DataFrame({'Dataset': dataset,
                                                          'MSE': squared_errors.mean(),
                                                          'p-value': p_value}, index)])

    print_uci_table(df_mse)


if __name__ == '__main__':

    # output directory
    if not os.path.exists('tables'):
        os.mkdir('tables')

    # UCI tables
    generate_uci_tables()
