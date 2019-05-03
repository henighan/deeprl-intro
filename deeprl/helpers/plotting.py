""" Plotters """
import os
import logging

from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats

from deeprl.utils import logdir
from deeprl.common import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def deeprlplot(exp_name, implementations, num_runs=None,
                value='AverageEpRet', epochs=None, benchmark=False, **kwargs):
    """ Plot results """
    if benchmark:
        fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(8,6))
        delta_eprets = pd.DataFrame()
    else:
        fig, ax0 = plt.subplots()
    for imp in implementations:
        data = get_dataframe(exp_name, imp, num_runs, value, epochs, kwargs)
        ax0.plot(data['TotalEnvInteracts'], data[value], '.', label=imp,
                 markersize=2, alpha=0.5)
        if benchmark:
            tmp = calculate_delta_ep_ret(data, epochs, value=value)
            tmp['imp'] = imp
            delta_eprets = pd.concat([delta_eprets, tmp])
    if benchmark:
        tom_inds = delta_eprets['imp']=='tom'
        t_stat, p_val = stats.ttest_ind(
            delta_eprets[tom_inds]['DeltaEpRet'],
            delta_eprets[~tom_inds]['DeltaEpRet'])
        sns.stripplot(x='imp', y='DeltaEpRet', data=delta_eprets, ax=ax1)
        ax1.set_title('p-val {}'.format(p_val))
        ax1.set_xlabel('implementation')
    ax0.legend()
    ax0.set_xlabel('TotalEnvInteracts')
    ax0.set_ylabel(value)
    ax0.set_title(str(kwargs))
    plt.tight_layout()
    plt.draw()
    plt.pause(0.001)


def get_dataframe(exp_name, imp, num_runs, value, epochs, kwargs):
    """ Get a pandas dataframe of the 'value' column from progress.txt logged
    by the spinup logger. Also gets the 'Epochs' and 'TotalEnvInteracts'
    columns. If epochs is specified, it will only return data up to that
    epoch. """
    data = pd.DataFrame()
    seeds = logdir.seeds(num_runs) if num_runs \
            else logdir.already_run_seeds(exp_name, imp, kwargs)
    for seed in seeds:
        output_dir = logdir.output_dir_from_kwargs(
            exp_name, imp, kwargs, seed=seed)
        logging.debug('plotter reading {}'.format(output_dir))
        path = os.path.join(output_dir, 'progress.txt')
        try:
            tmp_df = pd.read_csv(path, delimiter='\t')[[
                value, 'Epoch', 'TotalEnvInteracts']]
        except pd.errors.EmptyDataError:
            logging.debug('no data found in {}. \nSkipping'.format(
                output_dir))
            continue
        tmp_df['seed'] = seed
        if epochs:
            min_epoch = tmp_df['Epoch'].min()
            tmp_df['Epoch']
            tmp_df = tmp_df[tmp_df['Epoch'] < min_epoch + epochs]
        data = pd.concat([tmp_df, data])
    return data


def calculate_delta_ep_ret(data, epochs, value='AverageEpRet'):
    starting_epret = data[data['Epoch']==data['Epoch'].min()][[
        value, 'seed']].rename(
        {value: 'StartingEpRet'}, axis=1)
    ending_epret = data[data['Epoch']==data['Epoch'].max()][[
        value, 'seed']].rename(
        {value: 'EndingEpRet'}, axis=1)
    merged = pd.merge(starting_epret, ending_epret, on='seed')
    diff  = merged['EndingEpRet'] - merged['StartingEpRet']
    diff.name = 'DeltaEpRet'
    return pd.DataFrame(diff)
