import os
import logomaker
import pickle

import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

from matplotlib import pyplot as plt


def convergence_plots():
    data_file = os.path.join('experiments', 'convergence', 'data.pkl')
    metrics_file = os.path.join('experiments', 'convergence', 'metrics.pkl')
    measurements_file = os.path.join('experiments', 'convergence', 'measurements.pkl')
    if not os.path.exists(data_file) or not os.path.exists(metrics_file) or not os.path.exists(measurements_file):
        return

    # load data and measurements
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    metrics = pd.read_pickle(metrics_file)
    measurements = pd.read_pickle(measurements_file)

    # learning curve figure
    fig_learning_curve, ax = plt.subplots(ncols=2, figsize=(10, 5))
    sns.lineplot(data=metrics.reset_index(), x='Epoch', y='RMSE', hue='Model', ax=ax[0])
    sns.lineplot(data=metrics.reset_index(), x='Epoch', y='ECE', hue='Model', ax=ax[1])
    plt.tight_layout()
    fig_learning_curve.savefig(os.path.join('results', 'toy_learning_curve.pdf'))

    # convergence figure
    models = measurements.index.unique(0)
    fig_convergence, ax = plt.subplots(nrows=2, ncols=len(models), figsize=(5 * len(models), 10))
    for i, model in enumerate(models):

        # title
        ax[0, i].set_title(model)

        # plot data
        sizes = 12.5 * np.ones_like(data['x_train'])
        sizes[-2:] = 125
        ax[0, i].scatter(data['x_train'], data['y_train'], alpha=0.5, s=sizes)

        # predictive moments
        df = measurements.loc[model].reset_index()
        palette = 'ch:s=.3,rot=-.25'
        sns.lineplot(data=df, x='x', y='Mean', hue='Epoch', legend=False, palette=palette, ax=ax[0, i])
        sns.lineplot(data=df, x='x', y='Std. Deviation', hue='Epoch', legend=False, palette=palette, ax=ax[1, i])

        # true mean and standard deviation
        ax[0, i].plot(data['x_test'], data['target_mean'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)
        ax[1, i].plot(data['x_test'], data['target_std'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)

        # make things pretty
        ax[1, i].set_ylim([ax[1, i].get_ylim()[0], 8.5])

    # finalize and save figure
    plt.tight_layout()
    fig_convergence.savefig(os.path.join('results', 'toy_convergence.pdf'))


def analyze_performance(df_measurements, index, dataset, alpha=0.1, ece_bins=5, ece_method='one-sided'):

    # MSE
    squared_errors = df_measurements.loc[index, 'squared errors']
    null_index = 'Unit Variance Normal'
    if isinstance(index, pd.MultiIndex):
        null_index = index.set_levels([null_index], level='Model')
    null_squared_errors = df_measurements.loc[null_index, 'squared errors']
    mse = squared_errors.mean()
    p = stats.ttest_ind(squared_errors, null_squared_errors, equal_var=False, alternative='greater')[1]
    # p = stats.mannwhitneyu(squared_errors, null_squared_errors, alternative='greater')[1]
    mse = '\\sout{{{:.2g}}}'.format(mse) if p < alpha else '{:.2g}'.format(mse)
    p = '$H_0$' if index == null_index else '{:.2g}'.format(p)
    df_mse = pd.DataFrame({'Dataset': dataset, 'MSE': mse, 'Welch\'s $p$': p}, index)

    # ECE
    cdf_y = df_measurements.loc[index, 'F(y)'].to_numpy()
    p = np.stack([x / ece_bins for x in range(ece_bins + 1)])
    if ece_method == 'one-sided':
        p_hat = [sum(cdf_y <= p[i]) / len(cdf_y) for i in range(len(p))]
        ece = '{:.2g}'.format(np.sum((p - p_hat) ** 2))
    elif ece_method == 'two-sided':
        p_hat = [sum((p[i - 1] < cdf_y) & (cdf_y <= p[i])) / len(cdf_y) for i in range(1, len(p))]
        ece = '{:.2g}'.format(np.sum((1 / ece_bins - np.array(p_hat)) ** 2))
    else:
        raise NotImplementedError
    df_ece = pd.DataFrame({'Dataset': dataset, 'ECE': ece}, index)

    return df_mse, df_ece


def print_table(df, file_name, null_columns=None, highlight_min=False):

    # rearrange table for LaTeX
    df = df.set_index('Model')
    df_latex = df.melt(id_vars=['Dataset'], value_vars=df.columns[1:], ignore_index=False)
    df_latex = df_latex.reset_index()
    df_latex = df_latex.pivot(index='Dataset', columns=['Model', 'variable'], values='value')
    df_latex = pd.concat([df_latex.loc[:, ('Unit Variance Normal', null_columns or slice(None))],
                          df_latex[['Heteroscedastic Normal']],
                          df_latex[['Faithful Heteroscedastic Normal']]], axis=1)
    style = df_latex.style.hide(axis=1, names=True)
    if highlight_min:
        style = style.highlight_min(props='bfseries:;', axis=1)
    style.to_latex(
        buf=os.path.join('results', file_name),
        column_format='l' + ''.join(['|' + 'l' * len(df_latex[alg].columns) for alg in df_latex.columns.unique(0)]),
        hrules=True,
        # multicol_align='p{2cm}'
    )


def generate_uci_tables(normalized):
    if not os.path.exists(os.path.join('experiments', 'uci')):
        return

    # loop over datasets with predictions
    df_mse = pd.DataFrame()
    df_ece = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'uci')):
        measurements_file = os.path.join('experiments', 'uci', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            df_measurements = pd.read_pickle(measurements_file).sort_index()
            df_measurements = df_measurements[df_measurements['normalized'] == normalized]

            # drop index levels with just one unique value
            for level in df_measurements.index.names:
                if len(df_measurements.index.unique(level)) == 1:
                    df_measurements.set_index(df_measurements.index.droplevel(level), inplace=True)

            # loop over models and configurations
            for index in df_measurements.index.unique():
                if isinstance(index, tuple):
                    index = pd.MultiIndex.from_tuples(tuples=[index], names=df_measurements.index.names)
                else:
                    index = pd.Index(data=[index], name=df_measurements.index.names)

                # analyze performance
                df_mse_add, df_ece_add = analyze_performance(df_measurements, index, dataset)
                df_mse = pd.concat([df_mse, df_mse_add])
                df_ece = pd.concat([df_ece, df_ece_add])

    # print tables
    suffix = ('_normalized_' if normalized else '_')
    df_mse = df_mse.reset_index().set_index(df_mse.index.names[1:])
    df_ece = df_ece.reset_index().set_index(df_ece.index.names[1:])
    assert set(df_mse.index.unique()) == set(df_ece.index.unique())
    for i, index in enumerate(df_mse.index.unique()):
        config_str = ''
        for level in range(df_mse.index.nlevels):
            config_str += df_mse.index.names[level] + '_'
            index_str = index[level] if df_mse.index.nlevels > 1 else index
            config_str += ''.join(c for c in str(index_str) if c.isalnum() or c.isspace()).replace(' ', '_')
        print_table(df_mse.loc[[index]], file_name='uci_mse' + suffix + config_str + '.tex', null_columns=['MSE'])
        if not normalized:
            print_table(df_ece.loc[[index]], file_name='uci_ece' + suffix + config_str + '.tex', highlight_min=True)


def generate_crispr_tables():
    if not os.path.exists(os.path.join('experiments', 'crispr')):
        return

    # loop over datasets with predictions
    df_mse = pd.DataFrame()
    df_ece = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        measurements_file = os.path.join('experiments', 'crispr', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            df_measurements = pd.read_pickle(measurements_file).sort_index()

            # analyze each model's performance
            for index in df_measurements.index.unique():
                index = pd.Index(data=[index], name=df_measurements.index.names)
                df_mse_add, df_ece_add = analyze_performance(df_measurements, index, dataset)
                df_mse = pd.concat([df_mse, df_mse_add])
                df_ece = pd.concat([df_ece, df_ece_add])

    # print tables
    print_table(df_mse.reset_index(), file_name='crispr_mse.tex', null_columns=['MSE'])
    print_table(df_ece.reset_index(), file_name='crispr_ece.tex', highlight_min=True)


def crispr_motif_plots():
    if not os.path.exists(os.path.join('experiments', 'crispr')):
        return

    # loop over datasets with SHAP values
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        shap_file = os.path.join('experiments', 'crispr', dataset, 'shap.pkl')
        if os.path.exists(shap_file):
            df_shap = pd.read_pickle(shap_file).sort_index()

            # plot learned motifs for each model
            for model in df_shap.index.unique():
                fig, ax = plt.subplots(nrows=2, figsize=(15, 5))
                fig.suptitle(model + ' : ' + dataset)
                mean_shap = pd.DataFrame()
                std_shap = pd.DataFrame()
                for nt in ['A', 'C', 'G', 'T']:
                    mask = np.array(df_shap.loc[model, 'sequence'].apply(lambda seq: [s == nt for s in seq]).to_list())
                    mean_shap[nt] = (mask * np.array(df_shap.loc[model, 'mean'].to_list())).sum(0) / mask.sum(0)
                    std_shap[nt] = (mask * np.array(df_shap.loc[model, 'std'].to_list())).sum(0) / mask.sum(0)
                logomaker.Logo(mean_shap, flip_below=False, ax=ax[0])
                logomaker.Logo(std_shap, flip_below=False, ax=ax[1])
                ax[0].set_ylabel('SHAP of the Mean')
                ax[1].set_ylabel('SHAP of the Std. Dev.')
                limit = max(abs(np.array(list(ax[0].get_ylim()) + list(ax[1].get_ylim()))))
                ax[1].set_ylim([-limit, limit])
                ax[1].set_ylim([-limit, limit])
                plt.tight_layout()
                file_name = 'crispr_shap_' + dataset + '_' + model.lower().replace(' ', '-') + '.pdf'
                fig.savefig(os.path.join('results', file_name))


if __name__ == '__main__':

    # output directory
    if not os.path.exists('results'):
        os.mkdir('results')

    # convergence plots
    convergence_plots()

    # UCI tables
    generate_uci_tables(normalized=False)
    generate_uci_tables(normalized=True)

    # CRISPR tables and figures
    generate_crispr_tables()
    crispr_motif_plots()

    # show plots
    plt.show()
