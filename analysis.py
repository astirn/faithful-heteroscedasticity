import argparse
import logomaker
import matplotlib
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

MEAN_ONLY = 'Unit Variance Homoscedastic'
HETEROSCEDASTIC_MODELS = (
    'Conventional Heteroscedastic',
    'Beta NLL (0.5)',
    'Beta NLL (1.0)',
    'VBEM*',
    'Proposal 1',
    'Proposal 2',
    'Faithful Heteroscedastic',
)
MODELS = (MEAN_ONLY,) + HETEROSCEDASTIC_MODELS
ABLATION_TESTS = ('Proposal 1', 'Proposal 2')
COMPETITIVE_MODELS = ('Beta NLL (0.5)', 'Beta NLL (1.0)', 'Faithful Heteroscedastic')
CONVERGENCE_SUBTITLES = {
    'Normal': {
        'Unit Variance Homoscedastic': '\n(mean-only per definition 1)',
        'Conventional Heteroscedastic': '\n(Nix and Weigend, 1994)',
        'Beta NLL (0.5)': '\n(Seitzer et al., 2022)',
        'Beta NLL (1.0)': '\n(Seitzer et al., 2022)',
        'Proposal 1': '\n(Ours)',
        'Proposal 2': '\n(Ours)',
        'Faithful Heteroscedastic': '\n(Ours, Proposals 1 + 2)'},
    'Deep Ensemble': {
        'Unit Variance Homoscedastic': '\n(mean-only baseline)',
        'Conventional Heteroscedastic': '\n(Lakshminarayanan et al., 2017)',
        'Faithful Heteroscedastic': '\n(Ours, Proposals 1 + 2)'},
    'Monte Carlo Dropout': {
        'Unit Variance Homoscedastic': '\n(Gal and Ghahramani, 2016)',
        'Conventional Heteroscedastic': '\n(Kendall and Gal, 2017)',
        'Faithful Heteroscedastic': '\n(Ours, Proposals 1 + 2)'},
    'Student': {
        'Unit Variance Homoscedastic': '\n(mean-only baseline)',
        'Conventional Heteroscedastic': '\n(Takahashi et al., 2018)',
        'VBEM*': '\n(Stirn and Knowles, 2020)',
        'Faithful Heteroscedastic': '\n(Ours, Proposals 1 + 2)'},
}


def filter_model_class(df, model_class):
    return df.loc[df.index.get_level_values('Class') == model_class, :]


def drop_unused_index_levels(performance):

    # drop index levels with just one unique value
    for level in performance.index.names:
        if len(performance.index.unique(level)) == 1 and performance.index.nlevels > 1:
            performance.set_index(performance.index.droplevel(level), inplace=True)

    return performance


def find_best_model(candidates, df, measurements, max_or_min, alpha, test):

    # find the best of the candidates
    if max_or_min == 'max':
        i_best = df[df.index.isin(candidates)].idxmax()
    elif max_or_min == 'min':
        i_best = df[df.index.isin(candidates)].idxmin()
    else:
        raise NotImplementedError

    # identify all models that are statistically indistinguishable from the best
    best_models = candidates.to_list()
    for index in candidates:
        if test(measurements.loc[i_best], measurements.loc[index]) < alpha:
            best_models.remove(index)

    return best_models


def format_table_entries(series, best_models, unfaithful_models):

    # format numerical values, bold best values, and strikeout unfaithful models
    series = series.apply(lambda x: '{:.3g}'.format(x))
    series.loc[best_models] = series.loc[best_models].apply(lambda s: '\\textbf{{{:s}}}'.format(s))
    series.loc[unfaithful_models] = series.loc[unfaithful_models].apply(lambda s: '\\sout{{{:s}}}'.format(s))

    return series


def analyze_performance(measurements, dataset, alpha=0.05, ece_method='two-sided', z_scores=None):

    # RMSE
    rmse = measurements['squared errors'].groupby(level=measurements.index.names).mean() ** 0.5

    # identify unfaithful models
    unfaithful_models = []
    null_squared_errors = measurements.loc[(MEAN_ONLY,), 'squared errors']
    for index in measurements.index.unique():
        squared_errors = measurements.loc[index, 'squared errors']
        if stats.ttest_rel(squared_errors, null_squared_errors, alternative='greater')[1] < alpha:
            unfaithful_models += [index]

    # exclude any unfaithful model from our candidates list
    candidates = rmse[~rmse.index.isin(unfaithful_models) & ~(rmse.index.get_level_values('Model') == MEAN_ONLY)]
    candidates = candidates.index.unique()

    # finalize RMSE table
    best_rmse_models = find_best_model(candidates, rmse, measurements['squared errors'], 'min', alpha,
                                       lambda best, alt: stats.ttest_rel(best, alt, alternative='less')[1])
    df = format_table_entries(rmse, best_rmse_models, unfaithful_models).to_frame('RMSE')

    # loop over descending ECE bin sizes until we satisfy G-test requirements
    for ece_bins in list(range(1000, 100, -250)) + list(range(100, 10, -10)) + list(range(10, 2, -1)):

        # ECE and QQ weighted squared quantile errors
        histograms = pd.Series(index=pd.MultiIndex.from_tuples([], names=measurements.index.names), dtype=object)
        ece_values = pd.Series(index=pd.MultiIndex.from_tuples([], names=measurements.index.names), dtype=float)
        bin_probs = np.stack([x / ece_bins for x in range(ece_bins + 1)])
        qq_values = pd.Series(index=pd.MultiIndex.from_tuples([], names=measurements.index.names), dtype=float)
        quantiles = np.linspace(0.025, 0.975, num=96)
        normal_quantiles = stats.norm.ppf(q=quantiles)
        for index in measurements.index.unique():
            if not isinstance(index, tuple):
                index = (index,)

            # grab scores (and flatten for multidimensional  targets)
            if z_scores is None:
                scores = np.array(measurements.loc[index, 'z'].to_list()).reshape([-1])
            else:
                scores = z_scores[str(dict(zip(measurements.index.names, index)))].reshape([-1])

            # index for logging calibration values
            index = pd.MultiIndex.from_tuples([index], names=measurements.index.names)

            # ECE
            histogram = np.histogram(stats.norm.cdf(scores), bin_probs)[0]
            p_hat = histogram / len(scores)
            if ece_method == 'one-sided':
                errors = (bin_probs[1:] - np.cumsum(p_hat)) ** 2
            elif ece_method == 'two-sided':
                errors = (1 / ece_bins - p_hat) ** 2
            else:
                raise NotImplementedError
            histograms = pd.concat([histograms, pd.Series(histogram, index=index.repeat(len(histogram)))])
            ece_values = pd.concat([ece_values, pd.Series(errors, index=index.repeat(len(errors)))])

            # QQ weighted squared quantile errors
            scores_quantiles = np.quantile(scores, q=quantiles)
            sqe = (scores_quantiles - normal_quantiles) ** 2
            qq_values = pd.concat([qq_values, pd.Series(sqe, index=index.repeat(len(sqe)))])

        # make sure satisfy G-test requirements comfortably (5 counts per bin minimum)
        if histograms[histograms.index.get_level_values('Model').isin(COMPETITIVE_MODELS)].min() >= 20:
            break

    # finalize ECE table
    ece = ece_values.groupby(level=measurements.index.names).sum()
    best_ece_models = find_best_model(candidates, ece, histograms, 'min', alpha,
                                      test=lambda x, y: stats.power_divergence(x, y, lambda_='log-likelihood')[1])
    ece = format_table_entries(ece, best_ece_models, unfaithful_models).to_frame('ECE')
    df = df.join(ece)

    # finalize QQ RMSE table
    qq = qq_values.groupby(level=measurements.index.names).mean() ** 0.5
    best_qq_models = find_best_model(candidates, qq, qq_values, 'min', alpha,
                                     lambda best, alt: stats.ttest_rel(best, alt, alternative='less')[1])
    qq = format_table_entries(qq, best_qq_models, unfaithful_models).to_frame('QQ')
    df = df.join(qq)

    # log likelihoods
    ll = measurements['log p(y|x)'].groupby(level=measurements.index.names).mean()
    best_ll_models = find_best_model(candidates, ll, measurements['log p(y|x)'], 'max', alpha,
                                     lambda best, alt: stats.ks_2samp(best, alt, alternative='greater')[1])
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
    df_latex.loc[index, MEAN_ONLY] = '--'

    # style and save
    style = df_latex.style.hide(axis=1, names=True)
    col_fmt = 'l' * len(row_idx)
    col_fmt += ''.join(['|' + 'c' * len(df_latex[alg].columns) for alg in df_latex.columns.unique(0)])
    file_name = os.path.join('results', file_name)
    style.to_latex(buf=file_name, column_format=col_fmt, multicol_align='|c', hrules=True, multirow_align='t')


def print_tables(df, experiment, non_config_indices=('Model',)):
    if len(df) == 0:
        return

    # each configuration gets its own table
    if set(df.index.names) == set(non_config_indices):
        df['dummy'] = 'index'
        df.set_index('dummy', append=True, inplace=True)
    df.reset_index(non_config_indices, inplace=True)
    for config in df.index.unique():
        df_config = df.loc[config, :]
        [df_config.set_index(non_config_index, append=True, inplace=True) for non_config_index in non_config_indices]
        df_config = drop_unused_index_levels(df_config)

        # configure table rows and columns
        rows = ['Dataset'] + list(df_config.index.names)
        cols = [rows.pop(rows.index('Model'))]

        # print metric tables
        suffix = [] if df.index.nunique() == 1 else ['-'.join(config).replace(' ', '')]
        for metrics in [['RMSE', 'ECE', 'LL']]:
            file_name = '_'.join([experiment] + [m.lower() for m in metrics] + suffix) + '.tex'
            print_table(df_config[['Dataset'] + metrics].reset_index(), file_name=file_name, row_idx=rows, col_idx=cols)


def uci_tables(model_class, normalized=True):

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'uci')):
        measurements_file = os.path.join('experiments', 'uci', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            measurements = pd.read_pickle(measurements_file).sort_index()
            measurements = drop_unused_index_levels(filter_model_class(measurements, model_class))
            measurements = measurements[measurements['normalized'] == normalized]

            # analyze performance
            df = pd.concat([df, analyze_performance(measurements, dataset)])

    # print the tables
    print_tables(df, 'uci_' + model_class.replace(' ', '_'))


def vae_tables(model_class):

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        measurements_df_file = os.path.join('experiments', 'vae', dataset, 'measurements_df.pkl')
        measurements_dict_file = os.path.join('experiments', 'vae', dataset, 'measurements_dict.pkl')
        dataset = dataset.replace('_', '-')
        if os.path.exists(measurements_df_file) and os.path.exists(measurements_dict_file):
            measurements_df = filter_model_class(pd.read_pickle(measurements_df_file).sort_index(), model_class)
            with open(measurements_dict_file, 'rb') as f:
                measurements_dict = pickle.load(f)

            # analyze performance for each observation type and architecture
            index_names = measurements_df.index.names
            measurements_df = measurements_df.reset_index(['Model', 'Class']).sort_index()
            for index in measurements_df.index.unique():
                df_obs_arch = measurements_df.loc[index, :].reset_index().set_index(index_names)
                df = pd.concat([df, analyze_performance(df_obs_arch, dataset, z_scores=measurements_dict['Z'])])

    # print the tables
    print_tables(df, 'vae_' + model_class.replace(' ', '_'), non_config_indices=('Model', 'Observations'))


def crispr_tables(model_class):

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        measurements_file = os.path.join('experiments', 'crispr', dataset, 'measurements.pkl')
        if os.path.exists(measurements_file):
            measurements = pd.read_pickle(measurements_file).sort_index()
            measurements = drop_unused_index_levels(filter_model_class(measurements, model_class))

            # analyze performance for each observation type
            for observations in measurements.index.unique('Observations'):
                obs_performance = measurements[measurements.index.get_level_values('Observations') == observations]
                df = pd.concat([df, analyze_performance(obs_performance, dataset)])

    # print the tables
    print_tables(df, 'crispr_' + model_class.replace(' ', '_'), non_config_indices=('Model', 'Observations'))


def toy_convergence_plots(model_class):

    # ensure requisite files exist
    data_file = os.path.join('experiments', 'convergence', 'data.pkl')
    measurements_file = os.path.join('experiments', 'convergence', 'measurements.pkl')
    if not os.path.exists(data_file) or not os.path.exists(measurements_file):
        return

    # load data and measurements
    with open(data_file, 'rb') as f:
        data = pickle.load(f)
    measurements = pd.read_pickle(measurements_file).sort_index()
    measurements = drop_unused_index_levels(filter_model_class(measurements, model_class))
    assert measurements.index.names == ['Model'], 'multiple configurations per model is not supported'

    # convergence figure
    palette = sns.color_palette('ch:s=.3,rot=-.25', as_cmap=True)
    models_and_configs = measurements.index.unique()
    fig, ax = plt.subplots(nrows=2, ncols=len(models_and_configs), figsize=(5 * len(models_and_configs), 10))
    models = [model for model in MODELS if model in measurements.index]
    for i, model in enumerate(models):

        # title
        ax[0, i].set_title(model + CONVERGENCE_SUBTITLES[model_class][model])

        # plot data
        sizes = 12.5 * np.ones_like(data['x_train'])
        sizes[-2:] = 125
        ax[0, i].scatter(data['x_train'], data['y_train'], alpha=0.5, s=sizes)

        # mark data rich regions
        for r in range(ax.shape[0]):
            x_bounds = [data['x_train'][:-2].min(), data['x_train'][:-2].max()]
            ax[r, i].fill_between(x_bounds, 0, 1, color='grey', alpha=0.5, transform=ax[r, i].get_xaxis_transform())
            x_bounds = [data['x_train'].min() - 0.1, data['x_train'].min() + 0.1]
            ax[r, i].fill_between(x_bounds, 0, 1, color='grey', alpha=0.5, transform=ax[r, i].get_xaxis_transform())
            x_bounds = [data['x_train'].max() - 0.1, data['x_train'].max() + 0.1]
            ax[r, i].fill_between(x_bounds, 0, 1, color='grey', alpha=0.5, transform=ax[r, i].get_xaxis_transform())

        # predictive moments
        df = measurements.loc[model].reset_index()
        legend = 'brief' if i == len(models) - 1 else False
        sns.lineplot(data=df, x='x', y='Mean', hue='Epoch', legend=legend, palette=palette, ax=ax[0, i])
        sns.lineplot(data=df, x='x', y='Std. Deviation', hue='Epoch', legend=legend, palette=palette, ax=ax[1, i])

        # true mean and standard deviation
        ax[0, i].plot(data['x_test'], data['target_mean'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)
        ax[1, i].plot(data['x_test'], data['target_std'], alpha=0.5, color='black', linestyle=':', linewidth=3.5)
        ax[0, i].set_ylim([-14, 19])
        ax[1, i].set_ylim([0, 6])

        # clean up labels
        ax[0, i].set_xlabel('')
        if i > 0:
            ax[0, i].set_ylabel('')
            ax[1, i].set_ylabel('')

    # finalize and save figure
    ax[0, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')
    ax[1, -1].legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Epoch')
    plt.tight_layout()
    fig.savefig(os.path.join('results', 'toy_convergence_' + model_class.replace(' ', '_') + '.pdf'))


def vae_plots(model_class, examples_per_class=1):

    # utility function vae plots
    def concat_examples(output, indices=None):
        output = output if indices is None else output[indices]
        return np.squeeze(np.concatenate(np.split(output, indices_or_sections=output.shape[0]), axis=2))

    # loop over available example images
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        measurements_df_file = os.path.join('experiments', 'vae', dataset, 'measurements_df.pkl')
        measurements_dict_file = os.path.join('experiments', 'vae', dataset, 'measurements_dict.pkl')
        if os.path.exists(measurements_df_file) and os.path.exists(measurements_dict_file):
            measurements_df = filter_model_class(pd.read_pickle(measurements_df_file).sort_index(), model_class)
            with open(measurements_dict_file, 'rb') as f:
                measurements_dict = pickle.load(f)

            # make sure we support what was provided
            observations = ['clean', 'corrupt']
            assert set(measurements_df.index.unique('Observations')) == set(observations)
            assert len(measurements_df.index.unique('Model')) * len(measurements_df.index.unique('Observations')) == \
                   measurements_df.index.nunique()

            # randomly select some examples of each class to plot
            np.random.seed(args.seed)
            i_plot = np.zeros(shape=0, dtype=int)
            for k in np.sort(np.unique(measurements_dict['Class labels'])):
                i_class = np.where(measurements_dict['Class labels'] == k)[0]
                i_plot = np.concatenate([i_plot, np.random.choice(i_class, size=examples_per_class)], axis=0)

            # prepare performance plot
            models = set(MODELS).intersection(set(measurements_df.index.unique('Model')))
            models = [model for model in MODELS if (model in models) and (model not in ABLATION_TESTS)]
            fig, ax = plt.subplots(nrows=1 + len(models), ncols=2, figsize=(16, 2 * (1 + len(models))))

            # plot data
            for col, observation in enumerate(observations):
                x = concat_examples(measurements_dict['Data'][observation], i_plot)
                ax[0, col].imshow(x, cmap='gray_r', vmin=-5, vmax=5)
                ax[0, col].set_title(observation.capitalize() + ' Data')
                ax[0, col].set_xticks([])
                ax[0, col].set_yticks([])

            # plot each model's performance (but skip ablation tests)
            for index in measurements_df.index.unique():
                index_dict = dict(zip(measurements_df.index.names, index))
                if index_dict['Model'] in ABLATION_TESTS:
                    continue

                # gather moments
                mean = concat_examples(measurements_dict['Mean'][str(index_dict)], i_plot)
                std = concat_examples(measurements_dict['Std.'][str(index_dict)], i_plot) - 5

                # plot moments
                row = 1 + models.index(index_dict['Model'])
                col = observations.index(index_dict['Observations'])
                ax[row, col].imshow(np.concatenate([mean, std], axis=0), cmap='gray_r', vmin=-5, vmax=5)
                ax[row, col].set_title(index_dict['Model'])
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([mean.shape[0] // 2, 3 * mean.shape[0] // 2], ['Mean', 'Std.'])

            # finalize and save figures
            plt.tight_layout()
            fig.savefig(os.path.join('results', '_'.join(['vae', model_class, 'moments', dataset])) + '.pdf')

            # prepare variance decomposition plot
            models = set(HETEROSCEDASTIC_MODELS).intersection(set(measurements_df.index.unique('Model')))
            models = [model for model in HETEROSCEDASTIC_MODELS if (model in models) and (model not in ABLATION_TESTS)]
            fig, ax = plt.subplots(nrows=1 + len(models), figsize=(8, 1.25 * (1 + len(models))))
            x = concat_examples(measurements_dict['Noise variance']['corrupt']) ** 0.5
            ax[0].imshow(x, cmap='gray_r', vmin=0, vmax=5)
            ax[0].set_title('True $\\sqrt{\\mathrm{noise \\ variance}}$ per class')
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # plot each model's performance
            for model in models:
                index_clean = measurements_df.loc[([model], [model_class], ['clean']), :].index.unique()
                index_corrupt = measurements_df.loc[([model], [model_class], ['corrupt']), :].index.unique()
                assert len(index_clean) == len(index_corrupt) == 1, 'only one configuration per model is supported'
                index_clean = str(dict(zip(measurements_df.index.names, index_clean[0])))
                index_corrupt = str(dict(zip(measurements_df.index.names, index_corrupt[0])))

                # find average noise variance per class
                std_clean = measurements_dict['Std.'][index_clean]
                std_corrupt = measurements_dict['Std.'][index_corrupt]
                mean_std_noise = []
                for k in np.sort(np.unique(measurements_dict['Class labels'])):
                    i_class = np.where(measurements_dict['Class labels'] == k)[0]
                    std_noise = np.clip(std_corrupt[i_class] - std_clean[i_class], 0, np.inf).mean(axis=0)
                    mean_std_noise += [std_noise]
                mean_std_noise = np.concatenate(mean_std_noise, axis=1)

                # plot recovered noise variance
                row = 1 + models.index(model)
                ax[row].imshow(mean_std_noise, cmap='gray_r', vmin=0, vmax=5)
                ax[row].set_title(model)
                ax[row].set_xticks([])
                ax[row].set_yticks([])

            # finalize and save figures
            plt.tight_layout()
            fig.savefig(os.path.join('results', '_'.join(['vae', model_class, 'noise', dataset])) + '.pdf')


def crispr_motif_plots(model_class):

    def sequence_mask(df):
        return np.array(df.apply(lambda seq: [s == nt for s in seq]).to_list())

    def set_y_limits(axes):
        limits = (min([min(a.get_ylim()) for a in axes.flatten()]), max([max(a.get_ylim()) for a in axes.flatten()]))
        for axis in axes.flatten():
            axis.set_ylim(limits)

    # loop over datasets with SHAP values
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        shap_file = os.path.join('experiments', 'crispr', dataset, 'shap.pkl')
        mean_file = os.path.join('experiments', 'crispr', dataset, 'mean.pkl')
        if os.path.exists(shap_file) and os.path.exists(shap_file):
            shap = drop_unused_index_levels(filter_model_class(pd.read_pickle(shap_file).sort_index(), model_class))
            mean = drop_unused_index_levels(filter_model_class(pd.read_pickle(mean_file).sort_index(), model_class))
            mean = mean.groupby(['Model', 'Observations']).mean()
            shap['sequence'] = shap['sequence'].apply(lambda seq: seq.decode('utf-8'))
            shap = shap.droplevel('Fold')

            # compute SHAP values
            shap_plot = pd.DataFrame()
            for idx in shap.index.unique():
                for moment in ['mean', 'std']:
                    shap_values = dict(A=np.empty(0), C=np.empty(0), G=np.empty(0), T=np.empty(0))
                    for nt in ['A', 'C', 'G', 'T']:
                        mask = sequence_mask(shap.loc[idx, 'sequence'])
                        shap_values[nt] = np.array(shap.loc[idx, moment].to_list())
                        shap_values[nt] = (mask * shap_values[nt]).sum(0) / mask.sum(0)
                        shap_values[nt] += mean.loc[idx, moment] / len(shap_values[nt])
                    index = pd.MultiIndex.from_tuples([idx + (moment,)])
                    shap_plot = pd.concat([shap_plot, pd.DataFrame(shap_values, index.repeat(len(shap_values['A'])))])

            # make sure below assumptions hold
            assert shap.index.names == mean.index.names == ['Model', 'Observations']
            observations = ['replicates', 'means']
            assert set(shap.index.unique('Observations')) == set(observations)
            assert set(mean.index.unique('Observations')) == set(observations)

            # plot SHAP values for every model
            for model in shap.index.unique('Model'):
                fig, ax = plt.subplots(nrows=5, figsize=(10, 15))
                for i, moment in enumerate(['mean', 'std']):
                    for j, observation in enumerate(['replicates', 'means']):
                        row = 2 * i + j
                        shap_values = shap_plot.loc[(model, observation, moment), :].reset_index(drop=True)
                        logomaker.Logo(shap_values, flip_below=False, ax=ax[row])
                        moment_name = moment.replace('std', '$\\sqrt{\\mathrm{variance}}$')
                        title = 'SHAP of the estimated {:s} when trained on {:s}'.format(moment_name, observation)
                        ax[row].set_title(title)

                    # SHAP noise variance = SHAP trained on replicates - SHAP trained on means
                    if moment == 'std':
                        shap_delta = pd.DataFrame()
                        for nt in ['A', 'C', 'G', 'T']:
                            shap_delta[nt] = shap_plot.loc[(model, 'replicates', 'std'), nt].values
                            shap_delta[nt] -= shap_plot.loc[(model, 'means', 'std'), nt].values
                        logomaker.Logo(shap_delta, flip_below=False, ax=ax[-1])
                        ax[-1].set_title('SHAP of the estimated $\\sqrt{\\mathrm{noise \\ variance}}$')

                # finalize and save
                set_y_limits(ax[:2])
                set_y_limits(ax[2:])
                plt.tight_layout()
                model = model.replace(' ', '')
                fig.savefig(os.path.join('results', '_'.join(['crispr', model_class, dataset, model]) + '.pdf'))


if __name__ == '__main__':

    # parser arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='all', help='which experiment to analyze')
    parser.add_argument('--model_class', type=str, default='Normal', help='which model class to analyze')
    parser.add_argument('--seed', type=int, default=853211, help='random number seed for reproducibility')
    args = parser.parse_args()

    # make sure output directory exists
    os.makedirs('results', exist_ok=True)

    # set font sizes
    matplotlib.rcParams['pdf.fonttype'] = 42
    plt.rc('axes', labelsize=20, titlesize=20)
    plt.rc('figure', titlesize=25)
    plt.rc('legend', fontsize=15, title_fontsize=20)
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)

    # convergence experiment
    if args.experiment in {'all', 'convergence'} and os.path.exists(os.path.join('experiments', 'convergence')):
        toy_convergence_plots(args.model_class)

    # UCI experiments
    if args.experiment in {'all', 'uci'} and os.path.exists(os.path.join('experiments', 'uci')):
        uci_tables(args.model_class, normalized=True)
        # TODO: SHAP analysis

    # VAE experiments
    if args.experiment in {'all', 'vae'} and os.path.exists(os.path.join('experiments', 'vae')):
        vae_tables(args.model_class)
        vae_plots(args.model_class)

    # CRISPR tables and figures
    if args.experiment in {'all', 'crispr'} and os.path.exists(os.path.join('experiments', 'crispr')):
        crispr_tables(args.model_class)
        crispr_motif_plots(args.model_class)

    # show plots
    plt.show()
