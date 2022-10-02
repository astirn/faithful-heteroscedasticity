import argparse
import logomaker
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns

HOMOSCEDASTIC_MODELS = ('Unit Variance',)
BASELINE_HETEROSCEDASTIC_MODELS = ('Heteroscedastic', 'Beta NLL (0.5)', 'Beta NLL (1.0)')
OUR_HETEROSCEDASTIC_MODELS = ('Second Order Mean', 'Faithful Heteroscedastic')
HETEROSCEDASTIC_MODELS = BASELINE_HETEROSCEDASTIC_MODELS + OUR_HETEROSCEDASTIC_MODELS
MODELS = HOMOSCEDASTIC_MODELS + HETEROSCEDASTIC_MODELS


def drop_unused_index_levels(performance):

    # drop index levels with just one unique value
    for level in performance.index.names:
        if len(performance.index.unique(level)) == 1:
            performance.set_index(performance.index.droplevel(level), inplace=True)

    return performance


def find_best_model(candidates, df, measurements, max_or_min, test_values, alpha, test=stats.ttest_rel):

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
    series = series.apply(lambda x: '{:.3g}'.format(x))
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
            scores = z_scores[str(dict(zip(measurements.index.names, index)))].reshape([-1])

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
        wse = (scores_quantiles - normal_quantiles) ** 2
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
            df_config = df.loc[(slice(None),) * len(non_config_indices) + config, :]
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


def vae_tables():

    # loop over datasets with measurements
    df = pd.DataFrame()
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        measurements_df_file = os.path.join('experiments', 'vae', dataset, 'measurements_df.pkl')
        measurements_dict_file = os.path.join('experiments', 'vae', dataset, 'measurements_dict.pkl')
        if os.path.exists(measurements_df_file) and os.path.exists(measurements_dict_file):
            measurements_df = pd.read_pickle(measurements_df_file).sort_index()
            with open(measurements_dict_file, 'rb') as f:
                measurements_dict = pickle.load(f)

            # analyze performance for each observation type
            for observations in measurements_df.index.unique('Observations'):
                obs_perf = measurements_df[measurements_df.index.get_level_values('Observations') == observations]
                df = pd.concat([df, analyze_performance(obs_perf, dataset, z_scores=measurements_dict['Z'])])

    # print the tables
    print_tables(df, 'vae', non_config_indices=('Model', 'Observations'))


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


def vae_plots(examples_per_class=1):

    # utility function vae plots
    def concat_examples(output, indices=None):
        output = output if indices is None else output[indices]
        return np.squeeze(np.concatenate(np.split(output, indices_or_sections=output.shape[0]), axis=2))

    # loop over available example images
    for dataset in os.listdir(os.path.join('experiments', 'vae')):
        measurements_df_file = os.path.join('experiments', 'vae', dataset, 'measurements_df.pkl')
        measurements_dict_file = os.path.join('experiments', 'vae', dataset, 'measurements_dict.pkl')
        if os.path.exists(measurements_df_file) and os.path.exists(measurements_dict_file):
            measurements_df = pd.read_pickle(measurements_df_file).sort_index()
            with open(measurements_dict_file, 'rb') as f:
                measurements_dict = pickle.load(f)

            # make sure we support what was provided
            assert set(measurements_df.index.unique('Model')) == set(MODELS)
            observations = ['clean', 'corrupt']
            assert set(measurements_df.index.unique('Observations')) == set(observations)
            assert measurements_df.index.nunique() == len(MODELS) * len(observations)

            # randomly select some examples of each class to plot
            np.random.seed(args.seed)
            i_plot = np.zeros(shape=0, dtype=int)
            for k in np.sort(np.unique(measurements_dict['Class labels'])):
                i_class = np.where(measurements_dict['Class labels'] == k)[0]
                i_plot = np.concatenate([i_plot, np.random.choice(i_class, size=examples_per_class)], axis=0)

            # prepare performance plot
            fig, ax = plt.subplots(nrows=1 + len(MODELS), ncols=2, figsize=(15, 2 * (1 + len(MODELS))))
            fig.suptitle(dataset)

            # plot data
            for col, observation in enumerate(observations):
                x = concat_examples(measurements_dict['Data'][observation], i_plot)
                ax[0, col].imshow(x, cmap='gray_r', vmin=-0.5, vmax=0.5)
                ax[0, col].set_title(observation.capitalize() + ' Data')
                ax[0, col].set_xticks([])
                ax[0, col].set_yticks([])

            # plot each model's performance
            for index in measurements_df.index.unique():
                index_dict = dict(zip(measurements_df.index.names, index))

                # gather moments
                mean = concat_examples(measurements_dict['Mean'][str(index_dict)], i_plot)
                std = concat_examples(measurements_dict['Std.'][str(index_dict)], i_plot) - 0.5

                # plot moments
                row = 1 + MODELS.index(index_dict['Model'])
                col = observations.index(index_dict['Observations'])
                ax[row, col].imshow(np.concatenate([mean, std], axis=0), cmap='gray_r', vmin=-0.5, vmax=0.5)
                ax[row, col].set_title(index_dict['Model'])
                ax[row, col].set_xticks([])
                ax[row, col].set_yticks([mean.shape[0] // 2, 3 * mean.shape[0] // 2], ['Mean', 'Std.'])

            # finalize and save figures
            plt.tight_layout()
            fig.savefig(os.path.join('results', '_'.join(['vae', dataset, 'moments.pdf'])))

            # prepare variance decomposition plot
            fig, ax = plt.subplots(nrows=1 + len(HETEROSCEDASTIC_MODELS), figsize=(5, 5))
            x = concat_examples(measurements_dict['Noise variance']['corrupt']) ** 0.5
            ax[0].imshow(x, cmap='gray_r', vmin=0, vmax=1.0)
            ax[0].set_title('True $\\sqrt{\\mathrm{noise \\ variance}}$ per class')
            ax[0].set_xticks([])
            ax[0].set_yticks([])

            # plot each model's performance
            for model in HETEROSCEDASTIC_MODELS:
                index_clean = measurements_df.loc[([model], ['clean']), :].index.unique()
                index_corrupt = measurements_df.loc[([model], ['corrupt']), :].index.unique()
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
                row = 1 + HETEROSCEDASTIC_MODELS.index(model)
                ax[row].imshow(mean_std_noise, cmap='gray_r', vmin=0, vmax=1.0)
                ax[row].set_title(model)
                ax[row].set_xticks([])
                ax[row].set_yticks([])

            # finalize and save figures
            plt.tight_layout()
            fig.savefig(os.path.join('results', '_'.join(['vae', dataset, 'noise.pdf'])))


def crispr_motif_plots():

    def sequence_mask(df):
        return np.array(df.apply(lambda seq: [s == nt for s in seq]).to_list())

    def set_y_limits(axes):
        limits = (min([min(a.get_ylim()) for a in axes.flatten()]), max([max(a.get_ylim()) for a in axes.flatten()]))
        for axis in axes.flatten():
            axis.set_ylim(limits)

    # loop over datasets with SHAP values
    for dataset in os.listdir(os.path.join('experiments', 'crispr')):
        shap_file = os.path.join('experiments', 'crispr', dataset, 'shap.pkl')
        mean_output_file = os.path.join('experiments', 'crispr', dataset, 'mean_output.pkl')
        if os.path.exists(shap_file) and os.path.exists(shap_file):
            df_shap = drop_unused_index_levels(pd.read_pickle(shap_file).sort_index())
            df_mean_output = drop_unused_index_levels(pd.read_pickle(mean_output_file).sort_index())
            assert df_shap.index.names == ['Model', 'Observations']
            assert df_mean_output.index.names == ['Model', 'Observations']
            df_shap['sequence'] = df_shap['sequence'].apply(lambda seq: seq.decode('utf-8'))

            # make sure we support what was provided
            assert set(df_shap.index.unique('Model')) == set(MODELS)
            assert set(df_mean_output.index.unique('Model')) == set(MODELS)
            observations = ['replicates', 'means']
            assert set(df_shap.index.unique('Observations')) == set(observations)
            assert set(df_mean_output.index.unique('Observations')) == set(observations)
            assert df_shap.index.nunique() == len(MODELS) * len(observations)
            assert df_mean_output.index.nunique() == len(MODELS) * len(observations)

            # loop over the moments
            for moment in ['mean', 'std']:
                if moment == 'mean':
                    models = MODELS
                    delta_title = 'noise mean'
                else:
                    models = HETEROSCEDASTIC_MODELS
                    delta_title = '$\\sqrt{\\mathrm{noise \\ variance}}$'

                # SHAP figure
                fig, ax = plt.subplots(nrows=len(models), ncols=len(observations) + 1, figsize=(15, 10))
                fig.suptitle(dataset.capitalize())
                ax[0, 0].set_title('SHAP of the {:s} when trained on means'.format(moment))
                ax[0, 1].set_title('SHAP of the {:s} when trained on replicates'.format(moment))
                ax[0, 2].set_title('SHAP of the {:s}'.format(delta_title))
                for row, model in enumerate(models):
                    ax[row, 0].set_ylabel(model)

                    # SHAP values when trained on means and replicates
                    shap = dict(means=pd.DataFrame(), replicates=pd.DataFrame())
                    for observation in observations:
                        for nt in ['A', 'C', 'G', 'T']:
                            mask = sequence_mask(df_shap.loc[(model, observation), 'sequence'])
                            shap_values = np.array(df_shap.loc[(model, observation), moment].to_list())
                            shap[observation][nt] = (mask * shap_values).sum(0) / mask.sum(0)
                            # shap[observation][nt] += df_mean_output.loc[(model, observation), moment]
                        logomaker.Logo(shap[observation], flip_below=False, ax=ax[row, observations.index(observation)])

                    # SHAP noise variance = SHAP trained on replicates - SHAP trained on means
                    shap_delta = pd.DataFrame()
                    for nt in ['A', 'C', 'G', 'T']:
                        shap_delta[nt] = shap['replicates'][nt] - shap['means'][nt]
                    logomaker.Logo(shap_delta, flip_below=False, ax=ax[row, 2])

                # finalize and save
                set_y_limits(ax[:, :2])
                set_y_limits(ax[:, 2])
                plt.tight_layout()
                fig.savefig(os.path.join('results', '_'.join(['crispr', 'shap', dataset, moment]) + '.pdf'))


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
        vae_plots()

    # CRISPR tables and figures
    if args.experiment in {'all', 'crispr'} and os.path.exists(os.path.join('experiments', 'crispr')):
        crispr_tables()
        crispr_motif_plots()

    # show plots
    plt.show()
