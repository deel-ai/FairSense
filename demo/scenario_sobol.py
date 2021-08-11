import numpy as np
import pandas as pd

from libfairness.data_management.factory import create_fairness_problem
from libfairness.indices.sobol import compute_sobol
from tests.test_sensivity_indices import gaussian_data_generator
from libfairness.visualization.sobol_visu import visu_sobol

if __name__ == '__main__':
    # Setup
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    nsample = 5*10**2
    data_sample = 10**4
    bootstrap_size = 150

    # Data
    def func(x): return np.sum(x, axis=1)
    data = gaussian_data_generator(
        sigma12=0., sigma13=0., sigma23=0., N=data_sample)

    # Use Case
    my_problem = create_fairness_problem(inputs=data, function=func)
    compute_sobol(my_problem, n=nsample, bs=bootstrap_size)
    visu_sobol(my_problem)