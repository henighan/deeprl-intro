""" Plotters """
import os
import glob
from matplotlib import pyplot as plt
from deeprl import log_utils
import pandas as pd

def plot(exp_name, implementation, **kwargs):
    output_dir_base = log_utils.output_dir_base_from_kwargs(
        exp_name, implementation, kwargs)
    dfs = {}
    for seed_dir in glob.glob(output_dir_base + "_s*"):
        seed = int(seed_dir.split('s')[-1])
        dfs[seed] = pd.read_csv(
            os.path.join(seed_dir, 'progress.txt'), delimiter='\t')
    dfs[0]['AverageEpRet'].plot()
    plt.show()
