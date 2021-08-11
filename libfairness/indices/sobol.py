
from math import erf, sqrt

import numpy as np
import pandas as pd
from libfairness.fairness_problem import FairnessProblem
from numpy.linalg import cholesky, inv
from tqdm import tqdm


def __check_arg_sobol(fairness_problem, n, N, bs):
    """Check if the arguments needed to compute the Sobol indices are correct.
    Rectify when it is possible, raise an exception otherwise.

    Args:
        fairness_problem (FairnessProblem): Data of the fairness problem.
        n ([type]): [description]
        N ([type]): [description]
        bs ([type]): [description]

    Raises:
        ValueError: Inputs unknown.
        ValueError: Outputs unknown.
    """
    if fairness_problem.get_inputs() is None:
        raise ValueError("FairnessProblem.inputs is not set yet.")
    if fairness_problem.get_function() is None:
        raise ValueError("FairnessProblem.function is not set yet.")


def compute_sobol(fairness_problem: FairnessProblem, n=1000, N=None, bs=150):
    """Take a function and a dataset and compute the sobol indices.
    Set a dataframe with the sobol indice (columns) for each variable (rows) in FairnessProblem.result.

    Args:
        fairness_problem (FairnessProblem): The fairness problem to study.
        n (int, optional): [description]. Defaults to 1000.
        N ([type], optional): [description]. Defaults to None.
        bs (int, optional): [description]. Defaults to 150.
    """
    __check_arg_sobol(fairness_problem, n, N, bs)

    variable_names = None
    if isinstance(fairness_problem.get_inputs(), pd.DataFrame):
        variable_names = fairness_problem.get_inputs().columns
        x = x.values
    sobol_table = []
    for i in tqdm(range(bs)):
        sobol_table.append(compute_sobol_table(
            fairness_problem.get_function(), fairness_problem.get_inputs(), n=n, N=N))
    bootstrap_table = np.stack(sobol_table)
    fairness_problem.set_result(
        sobol_table_to_dataframe(bootstrap_table, variable_names))


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
        cols = cols + list(map(lambda x: x + "_inf", cols)) + \
            list(map(lambda x: x + "_sup", cols))
    else:
        data = sobol_table
    if variable_names is None:
        variable_names = ["X%i" % i for i in range(avg_indices.shape[0])]
    return pd.DataFrame(
        data,
        columns=cols,
        index=variable_names
    )


def compute_sobol_table(f, x, y=None, n=1000, N=None):
    """
    Compute all sobol indices for all variables
    Args:
        y:
        f: the function to analyze
        x: numpy array containing the data
        n: number of sample used to compute the sobol indices
        N: number of sample used to compute marginals

    Returns: a numpy array containing sobol indices (columns) for each variable (row)

    """
    if y is not None:
        mode = "on_error"
        x = np.hstack((y, x))
    else:
        mode = "on_value"
    nb_variables = x.shape[1]
    cov = empirical_cov(x)
    f_inv = compute_marginal_inv_cumul_dist(x, N)
    sobol_table = np.vectorize(lambda variable: sobol_indices_at_i(
        f, variable, mode, n, cov, f_inv), signature='()->(n)')(range(nb_variables))
    sobol_table[:, 2:] = np.roll(sobol_table[:, 2:], -1, axis=0)
    return sobol_table


def sobol_indices_at_i(f, variable_index, mode, n, cov, f_inv):
    """
    Compute the sobol indices for a specific variable. Attention for independent indices, the output is shifted: calling
    this function on variable i will yield sobol indices for i and sobol_indep indicies for i-1.

    Args:
        f: the function to analyze
        variable_index: the column number of the variable we want to analyze.
        n: number of sample to compute sobol indices.
        cov: covariance matrix of the data distribution.
        f_inv: the inverse cumulative distribution functions of the marginals of the data. It is an array of functions
        where the index indicate the variable.

    Returns: The computed sobol indices in the following order: "sobol", "sobol_total", "sobol_ind", "sobol_total_ind"

    """
    if mode == "on_value":
        # shift row and columns of cov to match the shift done on the Xi
        cov = np.hstack([cov[:, variable_index:], cov[:, :variable_index]])
        cov = np.vstack([cov[variable_index:, :], cov[:variable_index, :]])
    else:
        cov = np.hstack(
            [cov[:, variable_index:], cov[:, 1:variable_index, cov[:, 0]]])
        cov = np.vstack(
            [cov[variable_index:, :], cov[1:variable_index, :], cov[0, :]])

    # generate the znc
    znc = generator(np.eye(cov.shape[0]), n)
    zncbis = generator(np.eye(cov.shape[0]), n)
    if mode == "on_value":
        zncter = np.hstack((znc[:, [0]], zncbis[:, 1:]))
        zncquad = np.hstack((zncbis[:, :-1], znc[:, [-1]]))
    else:
        zncter = np.hstack((znc[:, [0, 1]], zncbis[:, 2:]))
        # possibly wrong !
        zncquad = np.hstack((znc[:, 0], zncbis[:, 1:-1], znc[:, [-1]]))

    # compute the L
    L = cholesky(cov)

    # compute zc
    zc = np.matmul(znc, np.matmul(inv(cholesky(empirical_cov(znc))), L.T))
    zcbis = np.matmul(zncbis, np.matmul(
        inv(cholesky(empirical_cov(zncbis))), L.T))
    zcter = np.matmul(zncter, np.matmul(
        inv(cholesky(empirical_cov(zncter))), L.T))
    zcquad = np.matmul(zncquad, np.matmul(
        inv(cholesky(empirical_cov(zncquad))), L.T))

    # shift back the columns to original order
    zc = reorder_cols(zc, variable_index, mode, inverse=True)
    zcbis = reorder_cols(zcbis, variable_index, mode, inverse=True)
    zcter = reorder_cols(zcter, variable_index, mode, inverse=True)
    zcquad = reorder_cols(zcquad, variable_index, mode, inverse=True)

    # apply marginals
    zc = apply_marginals(zc, f_inv)
    zcbis = apply_marginals(zcbis, f_inv)
    zcter = apply_marginals(zcter, f_inv)
    zcquad = apply_marginals(zcquad, f_inv)

    # compute Vhat
    if mode == "on_value":
        V = np.mean([np.var(f(zc)), np.var(f(zcbis)),
                    np.var(f(zcter)), np.var(f(zcquad))])
    else:
        V = np.mean([np.var(f(zc[:, 1:])), np.var(f(zcbis[:, 1:])),
                    np.var(f(zcter[:, 1:])), np.var(f(zcquad[:, 1:]))])

    # compute sobol indices
    return np.array([
        sobol_unnormalized(zc, zcbis, zcter, f, mode) / V,
        sobol_total_unnormalized(zcbis, zcter, f, mode) / (2 * V),
        sobol_ind_unnormalized(zc, zcbis, zcquad, f, mode) / V,
        sobol_total_ind_unnormalized(zcbis, zcquad, f, mode) / (2 * V)
    ]
    )


def apply_marginals(z_cond, F_inv):
    """
    Apply the computed marginal in order to get back to the original distribution.

    Args:
        z_cond: the conditionally distributed gaussian vector.
        F_inv: the inverse cumulative function of each marginal

    Returns: the transformed marginals.

    """
    xj_cols = []
    import matplotlib.pyplot as plt
    for j in range(z_cond.shape[1]):
        xj_cols.append(
            np.expand_dims(F_inv[j](phi(z_cond[:, j])), axis=1)
        )  # 2.6 step 4.
    return np.hstack(xj_cols)  # turn the list of columns into a matrix


def compute_marginal_inv_cumul_dist(data_x: np.ndarray.__class__, N=None):
    """
    Compute the inverse cumulative distribution of each marginal based on empirical data.

    Args:
        data_x: empirical data as numpy array. rows are observations.
        N: number of sample used to compute the marginals

    Returns: an array of function to compute each marginal. (index indicate which marginal)

    """
    if N is None:
        N = len(data_x)
    cum_dists_functions = list()
    for i in range(data_x.shape[1]):
        assert len(data_x) > 0
        x_sorted = np.array(data_x.copy()[:, i])
        x_sorted = x_sorted[np.random.choice(len(data_x), N)]
        x_sorted.sort()
        # cum_dists_functions.append(np.vectorize(lambda x: float(x_sorted.searchsorted(x)) / float(len(x_sorted))))
        cum_dists_functions.append(np.vectorize(
            lambda x: x_sorted[min(int(x*len(x_sorted)), len(x_sorted)-1)]))
    return cum_dists_functions


@np.vectorize
def phi(x):
    """
    Cumulative distribution function for the standard normal distribution
    """
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def reorder_cols(data, col_index, mode, inverse=False):
    """
    Shift the columns in order put the ith first.
    Args:
        data: the array we want to swap columns
        col_index: the index of the column to put at the first place
        inverse: if true execute the inverse operation ie. shift the first columns at index i

    Returns: the array with shifted columns

    Examples:

        ```
        _reorder_cols([1,2,3], 1, False) = [2,3,1]
        _reorder_cols([2,3,1], 1, True) = [1,2,3]
        ```

    """
    if inverse:
        col_index = - col_index
    if mode == 'on_value':
        return np.hstack([data[:, col_index:], data[:, :col_index]])
    else:
        return np.hstack([data[:, 0], data[:, col_index:], data[:, 1:col_index]])


def generator(cov, n=100):
    """
    Generate normally distributed batches
    Args:
        cov: covariance matrix of the distribution
        n: number of samples

    Returns: numpy array containing the samples

    """
    return np.random.multivariate_normal([0] * np.shape(cov)[0], cov, size=n)


def empirical_cov(data):
    """
    compute empirical covariance.
    Args:
        data: numpy array with the data.

    Returns: the covariance matrix of the data

    """
    return np.cov(np.transpose(data))


def sobol_unnormalized(zc, zcbis, zcter, f, mode):
    if mode == "on_error":
        def rf(x): return np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.multiply(rf(zc), (rf(zcter) - rf(zcbis))))


def sobol_total_ind_unnormalized(zcbis, zcquad, f, mode):
    if mode == "on_error":
        def rf(x): return np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.square(rf(zcquad) - rf(zcbis)))


def sobol_ind_unnormalized(zc, zcbis, zcquad, f, mode):
    if mode == "on_error":
        def rf(x): return np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.multiply(rf(zc), (rf(zcquad) - rf(zcbis))))


def sobol_total_unnormalized(zcbis, zcter, f, mode):
    if mode == "on_error":
        def rf(x): return np.abs(f(x[:, 1:]) - x[:, 0])
    else:
        rf = f
    return np.mean(np.square(rf(zcter) - rf(zcbis)))
