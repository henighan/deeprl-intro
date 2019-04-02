""" Plotters """
import os

from matplotlib import pyplot as plt
import pandas as pd

from deeprl import log_utils


def single_plot(exp_name, num_runs, implementation,
                value='AverageEpRet', epochs=None, **kwargs):
    """ Plot results for a single implementation """
    data = pd.DataFrame()
    for run_no in range(num_runs):
        seed = 10*run_no
        output_dir = log_utils.output_dir_from_kwargs(
            exp_name, implementation, kwargs, seed=seed)
        path = os.path.join(output_dir, 'progress.txt')
        tmp_df = pd.read_csv(path, delimiter='\t')[[
            value, 'Epoch', 'TotalEnvInteracts']]
        if epochs:
            tmp_df = tmp_df[tmp_df['Epoch'] < epochs]
        data = pd.concat([tmp_df, data])
    plt.plot(data['TotalEnvInteracts'], data[value], 'o')
    plt.xlabel('TotalEnvInteracts')
    plt.ylabel(value)
    plt.show()
