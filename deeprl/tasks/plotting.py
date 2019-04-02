""" Plotters """
import os
import logging

from matplotlib import pyplot as plt
import pandas as pd

from deeprl import log_utils
from deeprl.common import LOGGER_NAME

logger = logging.getLogger(LOGGER_NAME)


def deeprlplot(exp_name, implementations, num_runs=None,
                value='AverageEpRet', epochs=None, **kwargs):
    """ Plot results """
    for imp in implementations:
        data = pd.DataFrame()
        seeds = log_utils.seeds(num_runs) if num_runs \
                else log_utils.already_run_seeds(exp_name, imp, kwargs)
        for seed in seeds:
            output_dir = log_utils.output_dir_from_kwargs(
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
            if epochs:
                tmp_df = tmp_df[tmp_df['Epoch'] < epochs]
            data = pd.concat([tmp_df, data])
        plt.plot(data['TotalEnvInteracts'], data[value], 'o', label=imp)
    plt.xlabel('TotalEnvInteracts')
    plt.ylabel(value)
    plt.show()
