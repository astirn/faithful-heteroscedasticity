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
HETEROSCEDASTIC_MODELS = ('Heteroscedastic', 'Beta NLL-0.50', 'Beta NLL-1.00', 'Second Order Mean', 'Faithful Heteroscedastic')
MODELS = HOMOSCEDASTIC_MODELS + HETEROSCEDASTIC_MODELS


def drop_unused_index_levels(performance):

    # drop index levels with just one unique value
    for level in performance.index.names:
        if len(performance.index.unique(level)) == 1:
            performance.set_index(performance.index.droplevel(level), inplace=True)

    return performance


def find_best_model(candidates, df, measurements, max_or_min, test_values, alpha=0.1):

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
        if stats.ttest_rel(null_values, values, alternative=alternative)[1] < alpha:
            best_models.remove(index)

    return best_models


def format_table_entries(series, best_models, unfaithful_models):

    # format numerical values, bold best values, and strikeout unfaithful models
    series = series.apply(lambda x: '{:.2g}'.format(x))
    series.loc[best_models] = series.loc[best_models].apply(lambda s: '\\textbf{{{:s}}}'.format(s))
    series.loc[unfaithful_models] = series.loc[unfaithful_models].apply(lambda s: '\\sout{{{:s}}}'.format(s))

    return series


def analyze_performance(measurements, dataset, alpha=0.05, ece_bins=5, ece_method='one-sided'):

    # RMSE
    rmse = measurements['squared errors'].groupby(level=['Model', 'Architecture']).mean() ** 0.5

    # identify unfaithful models
    unfaithful_models = []
    null_squared_errors = measurements.loc[('Unit Variance', 'single'), 'squared errors']
    for index in measurements.index.unique():
        squared_errors = measurements.loc[index, 'squared errors']
        if stats.ttest_rel(squared_errors, null_squared_errors, alternative='greater')[1] < alpha:
            unfaithful_models += [index]

    # exclude any unfaithful model from our candidates list
    candidates = rmse[~rmse.index.isin(unfaithful_models) & ~rmse.index.isin([('Unit Variance', 'single')])]
    candidates = candidates.index.unique()

    # finalize RMSE table
    best_rmse_models = find_best_model(candidates, rmse, measurements, 'min', 'squared errors', alpha)
    rmse = format_table_entries(rmse, best_rmse_models, unfaithful_models).to_frame('RMSE')
    rmse['Dataset'] = dataset

    # QQ squared errors
    qq_squared_errors = pd.DataFrame()
    quantiles = np.linspace(0.05, 0.95)
    normal_quantiles = np.array([stats.norm.ppf(q=q) for q in quantiles])
    weights = np.array([stats.norm.pdf(stats.norm.ppf(q=q)) for q in quantiles])
    for index in measurements.index.unique():
        scores = np.array(measurements.loc[index, 'z'].to_list()).reshape([-1])
        scores_quantiles = np.array([np.quantile(scores, q=q) for q in quantiles])
        sqe = weights * (scores_quantiles - normal_quantiles) ** 2 / weights.sum()
        index = pd.MultiIndex.from_tuples([index], names=measurements.index.names).repeat(len(sqe))
        qq_squared_errors = pd.concat([qq_squared_errors, pd.DataFrame({'QQ squared errors': sqe}, index=index)])

    # finalize QQ squared errors
    qq = qq_squared_errors['QQ squared errors'].groupby(level=['Model', 'Architecture']).mean() ** 0.5
    best_qq_models = find_best_model(candidates, qq, qq_squared_errors, 'min', 'QQ squared errors', alpha)
    qq = format_table_entries(qq, best_qq_models, unfaithful_models).to_frame('QQ RMSE')
    qq['Dataset'] = dataset

    # log likelihoods
    ll = measurements['log p(y|x)'].groupby(level=['Model', 'Architecture']).mean()
    best_ll_models = find_best_model(candidates, ll, measurements, 'max', 'log p(y|x)', alpha)
    ll = format_table_entries(ll, best_ll_models, unfaithful_models).to_frame('LL')
    ll['Dataset'] = dataset

    # ECE
    ece = pd.DataFrame(index=measurements.index.unique())
    p = np.stack([x / ece_bins for x in range(ece_bins + 1)])
    for index in measurements.index.unique():
        cdf_y = measurements.loc[index, 'F(y|x)'].to_numpy()
        if ece_method == 'one-sided':
            p_hat = [sum(cdf_y <= p[i]) / len(cdf_y) for i in range(len(p))]
            ece.loc[index, 'ECE'] = np.sum((p - p_hat) ** 2)
        elif ece_method == 'two-sided':
            p_hat = [sum((p[i - 1] < cdf_y) & (cdf_y <= p[i])) / len(cdf_y) for i in range(1, len(p))]
            ece.loc[index, 'ECE'] = np.sum((1 / ece_bins - np.array(p_hat)) ** 2)
        else:
            raise NotImplementedError

    # mark the best models
    i_best = ece[ece.index.isin(candidates)].idxmin()

    # finalize ECE table
    ece.loc[unfaithful_models, 'ECE'] = ece.loc[unfaithful_models, 'ECE'].apply(lambda x: '\\sout{{{:.2g}}}'.format(x))
    ece.loc[i_best, 'ECE'] = ece.loc[i_best, 'ECE'].apply(lambda x: '\\textbf{{{:.2g}}}'.format(x))
    ece['Dataset'] = dataset

    return rmse, qq, ll


def print_table(df, file_name, row_idx=('Dataset',), col_idx=('Model',), models=MODELS):

    # rearrange table for LaTeX
    col_idx = [c for c in col_idx]
    df = df.set_index(col_idx)
    df_latex = df.melt(id_vars=row_idx, value_vars=[c for c in df.columns if c not in row_idx], ignore_index=False)
    df_latex = df_latex.reset_index()
    df_latex = df_latex.pivot(index=row_idx, columns=col_idx + ['variable'], values='value')
    df_latex = pd.concat([df_latex[[model]] for model in models if model in df_latex.columns.unique('Model')], axis=1)

    # compute total wins
    total_wins = '\\textit{{Total wins or ties}}'
    df_latex.loc[total_wins] = '0'
    for column in df_latex.columns:
        wins = df_latex[column].apply(lambda s: 'textbf' in s).sum()
        df_latex.loc[total_wins, column] = '\\textit{{{:d}}}'.format(wins)

    # style and save
    style = df_latex.style.hide(axis=1, names=True)
    col_fmt = 'l' * len(row_idx)
    col_fmt += ''.join(['|' + 'c' * len(df_latex[alg].columns) for alg in df_latex.columns.unique(0)])
    style.to_latex(buf=os.path.join('results', file_name), column_format=col_fmt, hrules=True, siunitx=True)


def toy_convergence_plots(heteroscedastic_architecture):

    # ensure requisite files exist
    data_file = os.path.join('experiments', 'convergence', 'data.pkl')
    opti_hist_file = os.path.join('experiments', 'convergence', 'optimization_history.pkl')
    measurements_file = os.path.join('experiments', 'convergence', 'measurements.pkl')
    if not os.path.exists(data_file) or not os.path.exists(opti_hist_file) or not os.path.exists(measurements_file):
        return

    # load data, metrics, and measurements
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    opti_hist = pd.read_pickle(opti_hist_file).reset_index()
    measurements = pd.read_pickle(measurements_file)

    # # learning curve figure
    # fig_learning_curve, ax = plt.subplots(ncols=2, figsize=(10, 5))
    # sns.lineplot(data=metrics, x='Epoch', y='RMSE', hue='Model', style='Model', ax=ax[0])
    # sns.lineplot(data=metrics, x='Epoch', y='ECE', hue='Model', style='Model', ax=ax[1])
    # plt.tight_layout()
    # fig_learning_curve.savefig(os.path.join('results', 'toy_learning_curve.pdf'))
    # learning curve
    # df = metrics[(metrics.Model == model) & (metrics.Architecture == architecture)]
    # ax[2, i].plot(df['Epoch'], df['RMSE'], color='tab:blue')
    # ax[2, i].set_xlabel('Epoch')
    # ax[2, i].set_ylabel('RMSE', color='tab:blue')
    # ax[2, i].set_ylim([0, None])
    # ax[2, i].tick_params(axis='y', labelcolor='tab:blue')
    # ax_twin = ax[2, i].twinx()
    # ax_twin.plot(df['Epoch'], df['ECE'], color='tab:orange')
    # ax_twin.set_ylabel('ECE', color='tab:orange')
    # ax_twin.set_ylim([0, None])
    # ax_twin.tick_params(axis='y', labelcolor='tab:orange')

    # convergence figure
    palette = sns.color_palette('ch:s=.3,rot=-.25', as_cmap=True)
    models = measurements.index.unique('Model')
    measurements.reset_index(inplace=True)
    fig, ax = plt.subplots(nrows=2, ncols=len(models), figsize=(5 * len(models), 10))
    fig.suptitle('Converge for {:s} architecture'.format(heteroscedastic_architecture))
    for i, model in enumerate(models):
        architecture = 'single' if model in HOMOSCEDASTIC_MODELS else heteroscedastic_architecture

        # title
        ax[0, i].set_title(model)

        # plot data and data rich region
        sizes = 12.5 * np.ones_like(data['x_train'])
        sizes[-2:] = 125
        ax[0, i].scatter(data['x_train'], data['y_train'], alpha=0.5, s=sizes)
        x_bounds = [data['x_train'][:-2].min(), data['x_train'][:-2].max()]
        ax[1, i].fill_between(x_bounds, 0, 1, color='grey', alpha=0.5, transform=ax[1, i].get_xaxis_transform())

        # predictive moments
        df = measurements[(measurements.Model == model) & (measurements.Architecture == architecture)]
        legend = 'full' if i == len(models) - 1 else False
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
    fig.savefig(os.path.join('results', 'toy_convergence_' + heteroscedastic_architecture + '.pdf'))


def uci_tables():

    # loop over datasets with measurements
    df_rmse = pd.DataFrame()
    df_qq = pd.DataFrame()
    df_ll = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'uci')):
        performance_file = os.path.join('experiments', 'uci', dataset, 'measurements.pkl')
        if os.path.exists(performance_file):
            performance = pd.read_pickle(performance_file)
            performance = performance[performance['normalized']]
            performance = drop_unused_index_levels(performance)

            # analyze performance
            df_rmse_dataset, df_qq_dataset, df_ll_dataset = analyze_performance(performance, dataset)
            df_rmse = pd.concat([df_rmse, df_rmse_dataset])
            df_qq = pd.concat([df_qq, df_qq_dataset])
            df_ll = pd.concat([df_ll, df_ll_dataset])

    # print tables
    rows = ['Dataset'] + list(df_rmse.index.names)
    cols = [rows.pop(rows.index('Model')), rows.pop(rows.index('Architecture'))]
    print_table(df_rmse.reset_index(), file_name='uci_rmse.tex', row_idx=rows, col_idx=cols)
    print_table(df_qq.reset_index(), file_name='uci_qq.tex', row_idx=rows, col_idx=cols)
    print_table(df_ll.reset_index(), file_name='uci_ll.tex', row_idx=rows, col_idx=cols)


def vae_tables(latent_dim=10):

    # loop over available measurements
    df_mse = pd.DataFrame()
    df_ece = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        performance_file = os.path.join('experiments', 'vae', dataset, str(latent_dim), 'performance.pkl')
        if os.path.exists(performance_file):
            performance = pd.read_pickle(performance_file).sort_index()
            performance = drop_unused_index_levels(performance)

            # analyze each model's performance
            for index in performance.index.unique():
                index = pd.MultiIndex.from_tuples([index], names=performance.index.names)
                null = index.set_levels([['Unit Variance'], ['single']], level=['Model', 'Architecture'])
                df_mse_add, df_ece_add = analyze_performance(performance, index, null, dataset.replace('_', '-'))
                df_mse = pd.concat([df_mse, df_mse_add])
                df_ece = pd.concat([df_ece, df_ece_add])

    # print tables
    rows = ['Dataset'] + list(df_mse.index.names)
    cols = [rows.pop(rows.index('Model')), rows.pop(rows.index('Architecture'))]
    print_table(df_mse.reset_index(), file_name='vae_mse.tex', print_cols='MSE', row_idx=rows, col_idx=cols)
    print_table(df_ece.reset_index(), file_name='vae_ece.tex', print_cols='ECE', row_idx=rows, col_idx=cols)


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


def crispr_tables():

    # loop over datasets with predictions
    df_mse = pd.DataFrame()
    df_ece = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        performance_file = os.path.join('experiments', 'crispr', dataset, 'performance.pkl')
        if os.path.exists(performance_file):
            performance = pd.read_pickle(performance_file).sort_index()
            performance.index = performance.index.droplevel('Fold')

            # analyze each model's performance
            for index in performance.index.unique():
                index = pd.MultiIndex.from_tuples([index], names=performance.index.names)
                null = index.set_levels([['Unit Variance'], ['single']], level=['Model', 'Architecture'])
                df_mse_add, df_ece_add = analyze_performance(performance, index, null, dataset)
                df_mse = pd.concat([df_mse, df_mse_add])
                df_ece = pd.concat([df_ece, df_ece_add])

    # print tables
    rows = ['Dataset'] + list(df_mse.index.names)
    cols = [rows.pop(rows.index('Model')), rows.pop(rows.index('Architecture'))]
    print_table(df_mse.reset_index(), file_name='crispr_mse.tex', print_cols='MSE', row_idx=rows, col_idx=cols)
    print_table(df_ece.reset_index(), file_name='crispr_ece.tex', print_cols='ECE', row_idx=rows, col_idx=cols)


def crispr_motif_plots(heteroscedastic_architecture='shared'):

    def sequence_mask(df):
        return np.array(df.apply(lambda seq: [s == nt for s in seq]).to_list())

    # loop over datasets with SHAP values
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        shap_file = os.path.join('experiments', 'crispr', dataset, 'shap.pkl')
        if os.path.exists(shap_file):
            df_shap = pd.read_pickle(shap_file)
            df_shap['sequence'] = df_shap['sequence'].apply(lambda seq: seq.decode('utf-8'))
            df_shap.index = df_shap.index.droplevel('Fold')
            df_shap = df_shap.loc[(slice(None), ['single', heteroscedastic_architecture], slice(None)), :]
            df_shap.index = df_shap.index.droplevel('Architecture')
            df_shap.sort_index(inplace=True)

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
        toy_convergence_plots(heteroscedastic_architecture='separate')
        toy_convergence_plots(heteroscedastic_architecture='shared')

    # UCI experiments
    if args.experiment in {'all', 'uci'} and os.path.exists(os.path.join('experiments', 'uci')):
        uci_tables()

    # VAE experiments
    if args.experiment in {'all', 'vae'} and os.path.exists(os.path.join('experiments', 'vae')):
        vae_tables()
        vae_plots(heteroscedastic_architecture='separate')
        vae_plots(heteroscedastic_architecture='shared')

    # CRISPR tables and figures
    if args.experiment in {'all', 'crispr'} and os.path.exists(os.path.join('experiments', 'crispr')):
        crispr_tables()
        crispr_motif_plots(heteroscedastic_architecture='separate')
        crispr_motif_plots(heteroscedastic_architecture='shared')

    # show plots
    plt.show()
