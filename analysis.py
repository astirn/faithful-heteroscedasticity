import argparse
import logomaker
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
import tensorflow as tf


def analyze_performance(df_measurements, index, dataset, alpha=0.1, ece_bins=5, ece_method='one-sided'):

    # MSE
    squared_errors = df_measurements.loc[index, 'squared errors']
    null_index = 'Unit Variance'
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


def print_table(df, file_name, row_cols=('Dataset',), null_columns=None, highlight_min=False):

    # rearrange table for LaTeX
    df = df.set_index('Model')
    df_latex = df.melt(id_vars=row_cols, value_vars=[c for c in df.columns if c not in row_cols], ignore_index=False)
    df_latex = df_latex.reset_index()
    df_latex = df_latex.pivot(index=row_cols, columns=['Model', 'variable'], values='value')
    df_latex = pd.concat([df_latex.loc[:, ('Unit Variance', null_columns or slice(None))],
                          df_latex[['Heteroscedastic']],
                          df_latex[['Faithful Heteroscedastic']]], axis=1)
    style = df_latex.style.hide(axis=1, names=True)
    if highlight_min:
        style = style.highlight_min(props='bfseries:;', axis=1)
    col_fmt = 'l' * len(row_cols)
    col_fmt += ''.join(['|' + 'l' * len(df_latex[alg].columns) for alg in df_latex.columns.unique(0)])
    style.to_latex(buf=os.path.join('results', file_name), column_format=col_fmt, hrules=True)
    # multicol_align='p{2cm}'


def toy_convergence_plots():
    data_file = os.path.join('experiments', 'convergence', 'data.pkl')
    metrics_file = os.path.join('experiments', 'convergence', 'metrics.pkl')
    measurements_file = os.path.join('experiments', 'convergence', 'measurements.pkl')
    if not os.path.exists(data_file) or not os.path.exists(metrics_file) or not os.path.exists(measurements_file):
        return

    # load data and measurements
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    metrics = pd.read_pickle(metrics_file).reset_index()
    measurements = pd.read_pickle(measurements_file)

    # learning curve figure
    fig_learning_curve, ax = plt.subplots(ncols=2, figsize=(10, 5))
    sns.lineplot(data=metrics, x='Epoch', y='RMSE', hue='Model', style='Model', ax=ax[0])
    sns.lineplot(data=metrics, x='Epoch', y='ECE', hue='Model', style='Model', ax=ax[1])
    plt.tight_layout()
    fig_learning_curve.savefig(os.path.join('results', 'toy_learning_curve.pdf'))

    # convergence figure
    palette = sns.color_palette('ch:s=.3,rot=-.25', as_cmap=True)
    models = measurements.index.unique(0)
    fig_convergence, ax = plt.subplots(nrows=2, ncols=len(models), figsize=(5 * len(models), 10))
    for i, model in enumerate(models):

        # title
        ax[0, i].set_title(model)

        # plot data and data rich region
        sizes = 12.5 * np.ones_like(data['x_train'])
        sizes[-2:] = 125
        ax[0, i].scatter(data['x_train'], data['y_train'], alpha=0.5, s=sizes)
        x_bounds = [data['x_train'][:-2].min(), data['x_train'][:-2].max()]
        ax[1, i].fill_between(x_bounds, 0, 1, color='grey', alpha=0.5, transform=ax[1, i].get_xaxis_transform())

        # predictive moments
        df = measurements.loc[model].reset_index()
        legend = 'full' if i == len(models) - 1 else False
        sns.lineplot(data=df, x='x', y='Mean', hue='Epoch', legend=legend, palette=palette, ax=ax[0, i])
        sns.lineplot(data=df, x='x', y='Std. Deviation', hue='Epoch', legend=legend, palette=palette, ax=ax[1, i])

        # true mean and standard deviation
        ax[0, i].plot(data['x_test'], data['target_mean'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)
        ax[1, i].plot(data['x_test'], data['target_std'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)

        # make things pretty
        ax[1, i].set_ylim([ax[1, i].get_ylim()[0], 8.5])

    # finalize and save figure
    ax[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')
    ax[1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')
    plt.tight_layout()
    fig_convergence.savefig(os.path.join('results', 'toy_convergence.pdf'))


def uci_tables(normalized):

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


def vae_plots():

    # loop over available measurements
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        for latent_dims in os.listdir(os.path.join('experiments', 'vae', dataset)):
            measurements_file = os.path.join('experiments', 'vae', dataset, latent_dims, 'test_examples.pkl')
            if os.path.exists(measurements_file):
                with open(measurements_file, 'rb') as f:
                    measurements = pickle.load(f)

                # loop over observation types
                for observation in ['clean', 'corrupt']:

                    # prepare plot and plot original data
                    fig, ax = plt.subplots(nrows=4, figsize=(10, 10))
                    x = tf.concat(tf.unstack(measurements['Data'][observation]), axis=1)
                    ax[0].imshow(x, cmap='gray_r', vmin=0, vmax=1)
                    ax[0].set_title('Data')
                    ax[0].set_xticks([])
                    ax[0].set_yticks([])

                    # loop over models
                    for i, model in enumerate(['Unit Variance', 'Heteroscedastic', 'Faithful Heteroscedastic']):
                        mean = tf.concat(tf.unstack(measurements['Mean'][observation][model]), axis=1)
                        std = tf.concat(tf.unstack(measurements['Std. deviation'][observation][model]), axis=1)
                        ax[i + 1].imshow(np.concatenate([mean, std], axis=0), cmap='gray_r', vmin=0, vmax=1)
                        ax[i + 1].set_title(model)
                        ax[i + 1].set_xticks([])
                        ax[i + 1].set_yticks([mean.shape[0] // 2, 3 * mean.shape[0] // 2], ['Mean', 'Std.'])

                    plt.tight_layout()


def crispr_tables():

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
    row_cols = ('Dataset', 'Observation')
    print_table(df_mse.reset_index(), row_cols=row_cols, file_name='crispr_mse.tex', null_columns=['MSE'])
    print_table(df_ece.reset_index(), row_cols=row_cols, file_name='crispr_ece.tex', highlight_min=True)


def crispr_motif_plots():

    def sequence_mask(df):
        return np.array(df.apply(lambda seq: [s == nt for s in seq]).to_list())

    # loop over datasets with SHAP values
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        shap_file = os.path.join('experiments', 'crispr', dataset, 'shap.pkl')
        if os.path.exists(shap_file):
            df_shap = pd.read_pickle(shap_file).sort_index()

            # models and observation order
            models = ['Unit Variance', 'Heteroscedastic', 'Faithful Heteroscedastic']
            observations = ['mean', 'replicates', 'mean']

            # SHAP of the mean figure
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
            fig.suptitle(dataset.capitalize())
            ax[0, 0].set_title('SHAP of the mean when trained on mean')
            ax[0, 1].set_title('SHAP of the mean when trained on replicates')
            for model, observation in df_shap.index.unique():
                shap = pd.DataFrame()
                for nt in ['A', 'C', 'G', 'T']:
                    mask = sequence_mask(df_shap.loc[(model, observation), 'sequence'])
                    shap_values = np.array(df_shap.loc[(model, observation), 'mean'].to_list())
                    shap[nt] = (mask * shap_values).sum(0) / mask.sum(0)
                ax_index = (models.index(model), observations.index(observation))
                logomaker.Logo(shap, flip_below=False, ax=ax[ax_index])
                ax[ax_index[0], 0].set_ylabel(model)
            y_limit = ax[0, 0].get_ylim()
            for ax in ax.flatten():
                ax.set_ylim(y_limit)
            plt.tight_layout()
            fig.savefig(os.path.join('results', 'crispr_shap_mean_' + dataset + '.pdf'))

            # SHAP of the standard deviation figure
            models.pop(0)
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
            fig.suptitle(dataset.capitalize())
            ax[0, 0].set_title(models[0])
            ax[0, 1].set_title(models[1])
            for col, model in enumerate(models):
                shap = dict(mean=pd.DataFrame(), replicates=pd.DataFrame())
                for observation in observations:
                    for nt in ['A', 'C', 'G', 'T']:
                        mask = sequence_mask(df_shap.loc[(model, observation), 'sequence'])
                        shap_values = np.array(df_shap.loc[(model, observation), 'std'].to_list())
                        shap[observation][nt] = (mask * shap_values).sum(0) / mask.sum(0)
                    ax_index = (observations.index(observation), col)
                    logomaker.Logo(shap[observation], flip_below=False, ax=ax[ax_index])
                    ax[ax_index[0], 0].set_ylabel('SHAP of the std. dev.\nwhen trained on ' + observation)
                shap_delta = pd.DataFrame()
                for nt in ['A', 'C', 'G', 'T']:
                    shap_delta[nt] = shap['replicates'][nt] - shap['mean'][nt]
                logomaker.Logo(shap_delta, flip_below=False, ax=ax[2, col])
                ax[2, col].set_ylabel('SHAP of the $\\sqrt{\\mathrm{noise \\ variance}}$')
            limit = max([max(abs(np.array(a.get_ylim()))) for a in ax.flatten()])
            for ax in ax.flatten():
                ax.set_ylim([-limit, limit])
            plt.tight_layout()
            fig.savefig(os.path.join('results', 'crispr_shap_std_' + dataset + '.pdf'))

            # # plot learned motifs for each model
            # for model in df_shap.index.unique():
            #     fig, ax = plt.subplots(nrows=2, figsize=(15, 5))
            #     fig.suptitle(model + ' : ' + dataset)
            #     mean_shap = pd.DataFrame()
            #     std_shap = pd.DataFrame()
            #     for nt in ['A', 'C', 'G', 'T']:
            #         mask = np.array(df_shap.loc[model, 'sequence'].apply(lambda seq: [s == nt for s in seq]).to_list())
            #         mean_shap[nt] = (mask * np.array(df_shap.loc[model, 'mean'].to_list())).sum(0) / mask.sum(0)
            #         std_shap[nt] = (mask * np.array(df_shap.loc[model, 'std'].to_list())).sum(0) / mask.sum(0)
            #     logomaker.Logo(mean_shap, flip_below=False, ax=ax[0])
            #     logomaker.Logo(std_shap, flip_below=False, ax=ax[1])
            #     ax[0].set_ylabel('SHAP of the Mean')
            #     ax[1].set_ylabel('SHAP of the Std. Dev.')
            #     limit = max(abs(np.array(list(ax[0].get_ylim()) + list(ax[1].get_ylim()))))
            #     ax[1].set_ylim([-limit, limit])
            #     ax[1].set_ylim([-limit, limit])
            #     plt.tight_layout()
            #     file_name = 'crispr_shap_' + dataset + '_' + model.lower().replace(' ', '-') + '.pdf'
            #     fig.savefig(os.path.join('results', file_name))


if __name__ == '__main__':

    # parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all', help='which experiment to analyze')
    args = parser.parse_args()

    # make sure output directory exists
    os.makedirs('results', exist_ok=True)

    # convergence experiment
    if args.experiment in {'all', 'convergence'} and os.path.exists(os.path.join('experiments', 'convergence')):
        toy_convergence_plots()

    # UCI experiments
    if args.experiment in {'all', 'uci'} and os.path.exists(os.path.join('experiments', 'uci')):
        uci_tables(normalized=False)
        uci_tables(normalized=True)

    # VAE experiments
    if args.experiment in {'all', 'vae'} and os.path.exists(os.path.join('experiments', 'vae')):
        # vae_tables()
        vae_plots()

    # CRISPR tables and figures
    if args.experiment in {'all', 'crispr'} and os.path.exists(os.path.join('experiments', 'crispr')):
        crispr_tables()
        crispr_motif_plots()

    # show plots
    plt.show()
