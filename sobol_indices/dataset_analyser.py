import pandas as pd
import numpy as np
from sobol_indices.ic_sampler import compute_sobol_table
from tqdm import tqdm


def sobol_table_to_dataframe(sobol_table, variable_names=None):
    """
    Turn the numpy array into a printable dataframe. Displays confidence intervals when bootstrapping.

    Args:
        sobol_table: numpy array containing the indices. Can be either 3D (bs, variable, index) when bootstrapping or 2D
        (variable, index) when not.
        variable_names: a list with the names of each variable. if None, the variables will be named X1, X2...

    Returns: a dataframe with the sobol indices inside.

    """
    cols = ["S", "ST", "S_ind", "ST_ind"]
    if len(sobol_table.shape) > 2:
        avg_indices = np.round(np.mean(sobol_table, axis=0), 2)
        avg_indices = np.vectorize(lambda x: min(1., max(0., x)))(avg_indices)
        inf_indices = np.round(np.percentile(sobol_table, 5, axis=0), 2)
        inf_indices = np.vectorize(lambda x: min(1., max(0., x)))(inf_indices)
        sup_indices = np.round(np.percentile(sobol_table, 95, axis=0), 2)
        sup_indices = np.vectorize(lambda x: min(1., max(0., x)))(sup_indices)
        # ci_sobol_table = np.vectorize(lambda inf, sup: "[%.2f,%.2f]" % (inf, sup))(inf_indices, sup_indices)
        data = np.hstack([avg_indices, inf_indices, sup_indices])
        cols = cols + list(map(lambda x: x + "_inf", cols)) + list(map(lambda x: x + "_sup", cols))
    else:
        data = sobol_table
    if variable_names is None:
        variable_names = ["X%i" % i for i in range(avg_indices.shape[0])]
    return pd.DataFrame(
        data,
        columns=cols,
        index=variable_names
    )


def analyze(f, x, y=None, n=1000, N=None, bs=150):
    """
    Take a function and a dataset and compute the sobol indices.
    Args:
        y:
        f: the function to analyze. The function must be vectorized ( must work with array of inputs )
        x: the dataset to analyze can be either a numpy array or a pandas dataframe.
        n: number of sample used to compute the indices
        bs: bootstrapping, number of runs used to compute confidence intervals.
        N: number of samples used to compute the marginals

    Returns: a dataframe with the sobol indice (columns) for each variable (rows)

    """
    variable_names = None
    if isinstance(x, pd.DataFrame):
        variable_names = x.columns
        x = x.values
    sobol_table = []
    for i in tqdm(range(bs)):
        sobol_table.append(compute_sobol_table(f, x, n=n, N=N))
    bootstrap_table = np.stack(sobol_table)
    return sobol_table_to_dataframe(bootstrap_table, variable_names)
