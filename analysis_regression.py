import os

import pandas as pd

from data_regression import REGRESSION_DATA
from utils import build_table, champions_club_table

if not os.path.exists('tables'):
    os.mkdir('tables')

results = dict()
for dataset in REGRESSION_DATA.keys():
    results_file = os.path.join('experiments', 'regression', dataset, 'results.pkl')
    if os.path.exists(results_file):
        df_results = pd.read_pickle(results_file)
        df_results = df_results[df_results.Model.isin({
            'HomoscedasticNormal',
            'HeteroscedasticNormal',
            'FaithfulHeteroscedasticNormal',
        })]
        results.update({dataset: df_results})

# make latex tables
max_cols = 5
champions_club = []
for metric in ['LL', 'RMSE', 'ECE']:
    order = 'max' if metric == 'LL' else 'min'
    with open(os.path.join('tables', 'regression_uci_' + metric.lower().replace(' ', '_') + '.tex'), 'w') as f:
        table, cc = build_table(results, metric, order, max_cols, bold_statistical_ties=True)
        print(table.replace('NaN', '--'), file=f)
    champions_club.append(cc)


# print champions club
with open(os.path.join('tables', 'regression_uci_champions_club.tex'), 'w') as f:
    print(champions_club_table(champions_club), file=f)