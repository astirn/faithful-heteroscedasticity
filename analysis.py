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

HOMOSCEDASTIC_MODELS = ('Unit Variance',)
BASELINE_HETEROSCEDASTIC_MODELS = ('Heteroscedastic', 'Beta NLL (0.5)', 'Beta NLL (1.0)')
OUR_HETEROSCEDASTIC_MODELS = ('Second Order Mean', 'Faithful Heteroscedastic')
HETEROSCEDASTIC_MODELS = BASELINE_HETEROSCEDASTIC_MODELS + OUR_HETEROSCEDASTIC_MODELS
MODELS = HOMOSCEDASTIC_MODELS + HETEROSCEDASTIC_MODELS
ARCHITECTURES = ('single', 'separate', 'shared')


def drop_unused_index_levels(performance):

    # drop index levels with just one unique value
    for level in performance.index.names:
        if len(performance.index.unique(level)) == 1:
            performance.set_index(performance.index.droplevel(level), inplace=True)

    return performance


def find_best_model(candidates, df, measurements, max_or_min, test_values, alpha=0.1, test=stats.ttest_rel):

    # find the best of the candidates
    if max_or_min == 'max':
        i_best = df[df.index.isin(candidates)].idxmax()
        alternative = 'greater'
    elif max_or_min == 'min':
        i_best = df[df.index.isin(candidates)].idxmin()
        alternative = 'less'
    else:
        raise NotImplementedError

    # identify all models that are statistically indistinguishable from the best
    best_models = candidates.to_list()
    for index in candidates:
        values = measurements.loc[index, test_values]
        null_values = measurements.loc[i_best, test_values]
        if test(null_values, values, alternative=alternative)[1] < alpha:
            best_models.remove(index)

    return best_models


def format_table_entries(series, best_models, unfaithful_models):

    # format numerical values, bold best values, and strikeout unfaithful models
    series = series.apply(lambda x: '{:.2g}'.format(x))
    series.loc[best_models] = series.loc[best_models].apply(lambda s: '\\textbf{{{:s}}}'.format(s))
    series.loc[unfaithful_models] = series.loc[unfaithful_models].apply(lambda s: '\\sout{{{:s}}}'.format(s))

    return series


def analyze_performance(measurements, dataset, alpha=0.05, ece_bins=5, ece_method='two-sided', z_scores=None):

    # RMSE
    rmse = measurements['squared errors'].groupby(level=measurements.index.names).mean() ** 0.5

    # identify unfaithful models
    unfaithful_models = []
    null_squared_errors = measurements.loc[('Unit Variance',), 'squared errors']
    for index in measurements.index.unique():
        squared_errors = measurements.loc[index, 'squared errors']
        if stats.ttest_rel(squared_errors, null_squared_errors, alternative='greater')[1] < alpha:
            unfaithful_models += [index]

    # exclude any unfaithful model from our candidates list
    candidates = rmse[~rmse.index.isin(unfaithful_models) & ~(rmse.index.get_level_values('Model') == 'Unit Variance')]
    candidates = candidates.index.unique()

    # finalize RMSE table
    best_rmse_models = find_best_model(candidates, rmse, measurements, 'min', 'squared errors', alpha)
    df = format_table_entries(rmse, best_rmse_models, unfaithful_models).to_frame('RMSE')

    # ECE and QQ weighted squared quantile errors
    ece = pd.Series(index=measurements.index.unique(), dtype=float)
    bin_probs = np.stack([x / ece_bins for x in range(ece_bins + 1)])
    qq_wse = pd.DataFrame()
    quantiles = np.linspace(0.025, 0.975, num=96)
    normal_quantiles = stats.norm.ppf(q=quantiles)
    weights = stats.norm.pdf(stats.norm.ppf(q=quantiles))
    weights /= np.sum(weights)
    for index in measurements.index.unique():
        if not isinstance(index, tuple):
            index = (index,)

        # grab scores (and flatten for multidimensional  targets)
        if z_scores is None:
            scores = np.array(measurements.loc[index, 'z'].to_list()).reshape([-1])
        else:
            scores = z_scores[' '.join(index[:2])].numpy().reshape([-1])

        # ECE
        cdf = stats.norm.cdf(scores)
        if ece_method == 'one-sided':
            p_hat = [sum(cdf <= bin_probs[i]) / len(cdf) for i in range(len(bin_probs))]
            ece.loc[index] = np.sum((bin_probs - p_hat) ** 2)
        elif ece_method == 'two-sided':
            p_hat = np.histogram(cdf, bin_probs)[0] / len(cdf)
            ece.loc[index] = np.sum((1 / ece_bins - np.array(p_hat)) ** 2)
        else:
            raise NotImplementedError

        # QQ weighted squared quantile errors
        scores_quantiles = np.quantile(scores, q=quantiles)
        wse = weights * (scores_quantiles - normal_quantiles) ** 2
        index = pd.MultiIndex.from_tuples([index], names=measurements.index.names).repeat(len(wse))
        qq_wse = pd.concat([qq_wse, pd.DataFrame({'QQ WSE': wse}, index=index)])

    # finalize ECE table
    best_ece_models = [ece[ece.index.isin(candidates)].idxmin()]
    ece = format_table_entries(ece, best_ece_models, unfaithful_models).to_frame('ECE')
    df = df.join(ece)

    # finalize QQ RMSE table
    qq = qq_wse['QQ WSE'].groupby(level=measurements.index.names).mean() ** 0.5
    best_qq_models = find_best_model(candidates, qq, qq_wse, 'min', 'QQ WSE', alpha)
    qq = format_table_entries(qq, best_qq_models, unfaithful_models).to_frame('QQ')
    df = df.join(qq)

    # log likelihoods
    ll = measurements['log p(y|x)'].groupby(level=measurements.index.names).mean()
    best_ll_models = find_best_model(candidates, ll, measurements, 'max', 'log p(y|x)', alpha, test=stats.ks_2samp)
    ll = format_table_entries(ll, best_ll_models, unfaithful_models).to_frame('LL')
    df = df.join(ll)

    # assign the dataset
    df['Dataset'] = dataset

    return df


def print_table(df, file_name, row_idx=('Dataset',), col_idx=('Model',), models=MODELS):

    # rearrange table for LaTeX
    col_idx = [c for c in col_idx]
    df = df.set_index(col_idx)
    df_latex = df.melt(id_vars=row_idx, value_vars=[c for c in df.columns if c not in row_idx], ignore_index=False)
    df_latex = df_latex.reset_index()
    df_latex = df_latex.pivot(index=row_idx, columns=col_idx + ['variable'], values='value')
    df_latex = pd.concat([df_latex[[model]] for model in models if model in df_latex.columns.unique('Model')], axis=1)

    # compute total wins
    total_wins = []
    for column in df_latex.columns:
        total_wins.append('\\textit{{{:d}}}'.format(df_latex[column].apply(lambda s: 'textbf' in s).sum()))
    index = ('\\textit{{Total wins or ties}}',) + ('',) * (len(df_latex.index.names) - 1)
    if len(df_latex.index.names) == 1:
        index = index[0]
    df_latex.loc[index, :] = total_wins

    # style and save
    style = df_latex.style.hide(axis=1, names=True)
    col_fmt = 'l' * len(row_idx)
    col_fmt += ''.join(['|' + 'c' * len(df_latex[alg].columns) for alg in df_latex.columns.unique(0)])
    file_name = os.path.join('results', file_name)
    style.to_latex(buf=file_name, column_format=col_fmt, hrules=True, multirow_align='t', siunitx=True)


def print_tables(df, experiment, non_config_indices=('Model',)):
    if len(df) == 0:
        return
    assert df.index.names[0] == 'Model', '"Model" needs to be first index level'

    # each configuration gets its own table
    if df.index.names == non_config_indices:
        configurations = [tuple()]
    else:
        configurations = df.index.droplevel(non_config_indices).unique()
    for config in configurations:
        if isinstance(df.index, pd.MultiIndex):
            df_config = df.loc[(slice(None),) + config, :]
        else:
            df_config = df.loc[(slice(None),) + config]
        df_config = drop_unused_index_levels(df_config)

        # configure table rows and columns
        rows = ['Dataset'] + list(df_config.index.names)
        cols = [rows.pop(rows.index('Model'))]

        # print metric tables
        suffix = [] if len(configurations) == 1 else ['-'.join(config).replace(' ', '')]
        for metric in ['RMSE', 'ECE', 'QQ', 'LL']:
            file_name = '_'.join([experiment, metric.lower()] + suffix) + '.tex'
            print_table(df_config[['Dataset', metric]].reset_index(), file_name=file_name, row_idx=rows, col_idx=cols)


def uci_tables(normalized=True):

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'uci')):
        measurements_file = os.path.join('experiments', 'uci', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            measurements = drop_unused_index_levels(pd.read_pickle(measurements_file).sort_index())
            measurements = measurements[measurements['normalized'] == normalized]

            # analyze performance
            df = pd.concat([df, analyze_performance(measurements, dataset)])

    # print the tables
    print_tables(df, 'uci')


def vae_tables(heteroscedastic_architecture=None, latent_dim=10):

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        performance_df_file = os.path.join('experiments', 'vae', dataset, str(latent_dim), 'measurements_df.pkl')
        performance_dict_file = os.path.join('experiments', 'vae', dataset, str(latent_dim), 'measurements_dict.pkl')
        if os.path.exists(performance_df_file) and os.path.exists(performance_dict_file):
            performance_df = pd.read_pickle(performance_df_file).sort_index()
            if heteroscedastic_architecture is not None:
                keep = performance_df.index.get_level_values('Architecture').isin(['single', heteroscedastic_architecture])
                performance_df = performance_df[keep]
            with open(performance_dict_file, 'rb') as f:
                performance_dict = pickle.load(f)

            # analyze performance for each observation type
            for observations in performance_df.index.unique('Observations'):
                obs_performance = performance_df[performance_df.index.get_level_values('Observations') == observations]
                z_scores = performance_dict['Z'][observations]
                df = pd.concat([df, analyze_performance(obs_performance, dataset, z_scores=z_scores)])

    # print the tables
    print_tables(df, 'vae', heteroscedastic_architecture)  # TODO: add latent dim to file name


def crispr_tables():

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        measurements_file = os.path.join('experiments', 'crispr', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            measurements = drop_unused_index_levels(pd.read_pickle(measurements_file).sort_index())

            # analyze performance for each observation type
            for observations in measurements.index.unique('Observations'):
                obs_performance = measurements[measurements.index.get_level_values('Observations') == observations]
                df = pd.concat([df, analyze_performance(obs_performance, dataset)])

    # print the tables
    print_tables(df, 'crispr', non_config_indices=('Model', 'Observations'))


def toy_convergence_plots():

    # ensure requisite files exist
    data_file = os.path.join('experiments', 'convergence', 'data.pkl')
    measurements_file = os.path.join('experiments', 'convergence', 'measurements.pkl')
    if not os.path.exists(data_file) or not os.path.exists(measurements_file):
        return

    # load data and measurements
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    measurements = drop_unused_index_levels(pd.read_pickle(measurements_file).sort_index())
    assert measurements.index.names == ['Model'], 'multiple configurations per model is not supported'

    # convergence figure
    palette = sns.color_palette('ch:s=.3,rot=-.25', as_cmap=True)
    models_and_configs = measurements.index.unique()
    fig, ax = plt.subplots(nrows=2, ncols=len(models_and_configs), figsize=(5 * len(models_and_configs), 10))
    for i, model in enumerate(MODELS):
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
        legend = 'full' if i == len(MODELS) - 1 else False
        sns.lineplot(data=df, x='x', y='Mean', hue='Epoch', legend=legend, palette=palette, ax=ax[0, i])
        sns.lineplot(data=df, x='x', y='Std. Deviation', hue='Epoch', legend=legend, palette=palette, ax=ax[1, i])

        # true mean and standard deviation
        ax[0, i].plot(data['x_test'], data['target_mean'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)
        ax[1, i].plot(data['x_test'], data['target_std'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)
        ax[0, i].set_ylim([-15, 20])
        ax[1, i].set_ylim([0, 6])

    # finalize and save figure
    ax[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')
    ax[1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')
    plt.tight_layout()
    fig.savefig(os.path.join('results', 'toy_convergence.pdf'))


def vae_plots(heteroscedastic_architecture, latent_dim=10, examples_per_class=1):
    assert heteroscedastic_architecture in {'separate', 'shared'}

    # utility function vae plots
    def concat_examples(output, indices=None):
        output = output if indices is None else tf.gather(output, indices)
        return tf.concat(tf.unstack(output), axis=1)

    # loop over available example images
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        plot_dict = os.path.join('experiments', 'vae', dataset, str(latent_dim), 'plot_dictionary.pkl')
        latent_str = ' {:}-dimensional latent space'.format(latent_dim)
        if os.path.exists(plot_dict):
            with open(plot_dict, 'rb') as f:
                plot_dict = pickle.load(f)

            # randomly select some examples of each class to plot
            tf.keras.utils.set_random_seed(args.seed)
            i_plot = tf.zeros(shape=0, dtype=tf.int64)
            for k in tf.sort(tf.unique(plot_dict['Class labels'])[0]):
                i_class = tf.where(tf.equal(plot_dict['Class labels'], k))
                i_plot = tf.concat([i_plot, tf.random.shuffle(i_class)[:examples_per_class, 0]], axis=0)

            # prepare performance plot
            fig, ax = plt.subplots(nrows=1 + len(MODELS), ncols=2, figsize=(15, 2 * (1 + len(MODELS))))
            fig.suptitle(dataset.format(latent_dim))

            # loop over observation types
            for col, observation in enumerate(['clean', 'corrupt']):

                # plot data
                x = concat_examples(plot_dict['Data'][observation], i_plot)
                ax[0, col].imshow(x, cmap='gray_r', vmin=-0.5, vmax=0.5)
                ax[0, col].set_title(observation.capitalize() + ' Data')
                ax[0, col].set_xticks([])
                ax[0, col].set_yticks([])

                # plot each model's performance
                for i, model in enumerate(MODELS):
                    architecture = 'single' if i == 0 else heteroscedastic_architecture
                    key = model + ' ' + architecture
                    if key not in plot_dict['Mean'][observation].keys():
                        continue
                    mean = concat_examples(plot_dict['Mean'][observation][key], i_plot)
                    std = concat_examples(plot_dict['Std. deviation'][observation][key], i_plot) - 0.5
                    ax[i + 1, col].imshow(np.concatenate([mean, std], axis=0), cmap='gray_r', vmin=-0.5, vmax=0.5)
                    ax[i + 1, col].set_title(model + ' w/ ' + architecture + latent_str)
                    ax[i + 1, col].set_xticks([])
                    ax[i + 1, col].set_yticks([mean.shape[0] // 2, 3 * mean.shape[0] // 2], ['Mean', 'Std.'])

            # finalize and save figures
            plt.tight_layout()
            file_name = 'vae_' + heteroscedastic_architecture + '_' + dataset + '_moments.pdf'
            fig.savefig(os.path.join('results', file_name))

            # prepare variance decomposition plot
            fig, ax = plt.subplots(nrows=1 + len(HETEROSCEDASTIC_MODELS), figsize=(10, 5))
            x = concat_examples(plot_dict['Noise variance']['corrupt']) ** 0.5
            ax[0].imshow(x, cmap='gray_r', vmin=0, vmax=1.0)
            ax[0].set_title('True $\\sqrt{\\mathrm{noise \\ variance}}$ per class')
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # loop over heteroscedastic models
            for i, model in enumerate(HETEROSCEDASTIC_MODELS):

                # find average noise variance per class
                key = model + ' ' + heteroscedastic_architecture
                if (key not in plot_dict['Mean']['clean'].keys()) or (key not in plot_dict['Mean']['corrupt'].keys()):
                    continue
                std_clean = plot_dict['Std. deviation']['clean'][key]
                std_corrupt = plot_dict['Std. deviation']['corrupt'][key]
                mean_std_noise = []
                for k in tf.sort(tf.unique(plot_dict['Class labels'])[0]):
                    i_class = tf.squeeze(tf.where(tf.equal(plot_dict['Class labels'], k)))
                    std_noise = tf.gather(std_corrupt, i_class) - tf.gather(std_clean, i_class)
                    std_noise = tf.clip_by_value(std_noise, 0, np.inf)
                    mean_std_noise += [tf.reduce_mean(std_noise, axis=0)]
                mean_std_noise = tf.concat(mean_std_noise, axis=1)

                # plot recovered noise variance
                ax[i + 1].imshow(mean_std_noise, cmap='gray_r', vmin=0, vmax=1.0)
                ax[i + 1].set_title(model + ' w/ ' + heteroscedastic_architecture + latent_str)
                ax[i + 1].set_xticks([])
                ax[i + 1].set_yticks([])

            # finalize and save figures
            plt.tight_layout()
            file_name = 'vae_' + heteroscedastic_architecture + '_' + dataset + '_noise.pdf'
            fig.savefig(os.path.join('results', file_name))


def crispr_motif_plots(heteroscedastic_architecture='shared'):

    def sequence_mask(df):
        return np.array(df.apply(lambda seq: [s == nt for s in seq]).to_list())

    # loop over datasets with SHAP values
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        shap_file = os.path.join('experiments', 'crispr', dataset, 'shap.pkl')
        if os.path.exists(shap_file):
            df_shap = pd.read_pickle(shap_file)
            df_shap.index = df_shap.index.droplevel('Fold')
            df_shap = pd.concat([
                df_shap.loc[(slice(None), ['single'], slice(None)), :],
                df_shap.loc[(slice(None), [heteroscedastic_architecture], slice(None)), :],
            ])
            df_shap.index = df_shap.index.droplevel('Architecture')
            df_shap.sort_index(inplace=True)
            df_shap['sequence'] = df_shap['sequence'].apply(lambda seq: seq.decode('utf-8'))

            # models and observation order
            models = ['Unit Variance', 'Heteroscedastic', 'Faithful Heteroscedastic']
            observations = ['means', 'replicates']
            if not len(df_shap.index.unique()) == len(models) * len(observations):
                continue

            # SHAP of the mean figure
            fig, ax = plt.subplots(nrows=len(models), ncols=len(observations), figsize=(15, 10))
            fig.suptitle(dataset.capitalize())
            ax[0, 0].set_title('SHAP of the mean when trained on means')
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
            file_name = 'crispr_shap_' + dataset + '_mean_' + heteroscedastic_architecture + '.pdf'
            fig.savefig(os.path.join('results', file_name))

            # SHAP of the standard deviation figure
            models.pop(0)
            fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 10))
            fig.suptitle(dataset.capitalize())
            ax[0, 0].set_title(models[0])
            ax[0, 1].set_title(models[1])
            for col, model in enumerate(models):
                shap = dict(means=pd.DataFrame(), replicates=pd.DataFrame())
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
                    shap_delta[nt] = shap['replicates'][nt] - shap['means'][nt]
                logomaker.Logo(shap_delta, flip_below=False, ax=ax[2, col])
                ax[2, col].set_ylabel('SHAP of the $\\sqrt{\\mathrm{noise \\ variance}}$')
            for col in range(2):
                limit = max([max(abs(np.array(a.get_ylim()))) for a in ax[:, col]])
                for row in range(3):
                    ax[row, col].set_ylim([-limit, limit])
            plt.tight_layout()
            file_name = 'crispr_shap_' + dataset + '_std_' + heteroscedastic_architecture + '.pdf'
            fig.savefig(os.path.join('results', file_name))

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
    parser.add_argument('--seed', type=int, default=853211, help='random number seed for reproducibility')
    args = parser.parse_args()

    # make sure output directory exists
    os.makedirs('results', exist_ok=True)

    # convergence experiment
    if args.experiment in {'all', 'convergence'} and os.path.exists(os.path.join('experiments', 'convergence')):
        toy_convergence_plots()

    # UCI experiments
    if args.experiment in {'all', 'uci'} and os.path.exists(os.path.join('experiments', 'uci')):
        uci_tables(normalized=True)

    # VAE experiments
    if args.experiment in {'all', 'vae'} and os.path.exists(os.path.join('experiments', 'vae')):
        vae_tables()
        vae_tables(heteroscedastic_architecture='separate')
        vae_tables(heteroscedastic_architecture='shared')
        # vae_plots(heteroscedastic_architecture='separate')
        # vae_plots(heteroscedastic_architecture='shared')

    # CRISPR tables and figures
    if args.experiment in {'all', 'crispr'} and os.path.exists(os.path.join('experiments', 'crispr')):
        crispr_tables()
        # crispr_motif_plots(heteroscedastic_architecture='separate')
        # crispr_motif_plots(heteroscedastic_architecture='shared')

    # show plots
    plt.show()
