import numpy as np
import pandas as pd

from sobol_indices.dataset_analyser import analyze
from sobol_indices.test_sensivity_indices import gaussian_data_generator

if __name__ == '__main__':
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    nsample = 5*10**2
    data_sample = 10**4
    bootstrap_size = 150

    func = lambda x: np.sum(x, axis=1)
    data = gaussian_data_generator(sigma12=0., sigma13=0., sigma23=0., N=data_sample)
    indices_df = analyze(func, data, n=nsample, bs=bootstrap_size)
    print("sigma12=.5, sigma13=0., sigma23=0. | f(X) -> X_1 + X_2 + X_3")
    print(indices_df[["S", "ST", "S_ind", "ST_ind"]])

    func = lambda x: x[:, 0]
    data = gaussian_data_generator(sigma12=0.5, sigma13=0., sigma23=0., N=data_sample)
    indices_df = analyze(func, data, n=nsample, bs=bootstrap_size)
    print("sigma12=.5, sigma13=0., sigma23=0. | f(X) -> X_0")
    print(indices_df[["S", "ST", "S_ind", "ST_ind"]])

    func = lambda x: np.sum(x, axis=1)
    data = gaussian_data_generator(sigma12=.5, sigma13=.8, sigma23=0., N=data_sample)
    indices_df = analyze(func, data, n=nsample, bs=bootstrap_size)
    print("sigma12=.5, sigma13=.8, sigma23=0 | f(X) -> X_1 + X_2 + X_3")
    print(indices_df[["S", "ST", "S_ind", "ST_ind"]])

    func = lambda x: np.sum(x, axis=1)
    data = gaussian_data_generator(sigma12=-0.5, sigma13=.2, sigma23=-0.7, N=data_sample)
    indices_df = analyze(func, data, n=nsample, bs=bootstrap_size)
    print("sigma12=-0.5, sigma13=.2, sigma23=-0.7 | f(X) -> X_1 + X_2 + X_3")
    print(indices_df[["S", "ST", "S_ind", "ST_ind"]])