from math import erf, sqrt
import numpy as np
import pandas as pd
from numpy.linalg import cholesky, inv
from fairsense.utils.dataclasses import IndicesInput, IndicesOutput


def sobol_indices(inputs: IndicesInput, n=1000, N=None) -> IndicesOutput:
    """
    Compute all sobol indices for all variables

    Warning:
        this indice may fail silently if all values of one variable are similar (
        constant ) which  may occurs when applying one hot encoding with a large
        number of splits.

    Args:
        inputs (IndicesInput): The fairness problem to study.
        n: number of sample used to compute the sobol indices
        N: number of sample used to compute marginals

    Returns:
        IndicesOutput object, containing the CVM indices, one line per variable group
        and one column for each index.

    """
    x = inputs.x
    cov = (x + np.random.normal(scale=1e-5, size=x.shape)).cov()
    orig_cols = []
    for group in inputs.variable_groups:
        orig_cols += group
    f_inv = _compute_marginal_inv_cumul_dist(x[orig_cols].values, N)
    sobol_table = []
    for i in range(len(inputs.variable_groups)):
        sobol_table.append(
            _sobol_indices_at_i(
                inputs.compute_objective, i, inputs.variable_groups, n, cov, f_inv
            )
        )
    sobol_table = np.vstack(sobol_table)
    sobol_table[:, 2:] = np.roll(sobol_table[:, 2:], -1, axis=0)
    return IndicesOutput(
        pd.DataFrame(
            data=sobol_table,
            index=inputs.merged_groups,
            columns=["S", "ST", "S_ind", "ST_ind"],
        )
    )


def _sobol_indices_at_i(f, variable_index, variable_groups, n, cov, f_inv):
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
    # global mx
    orig_cols = []
    for group in variable_groups:
        orig_cols += group
    reordered_cols = []
    for group in variable_groups[variable_index:] + variable_groups[:variable_index]:
        reordered_cols += group
    cov = cov[reordered_cols].loc[reordered_cols]

    # # shift row and columns of cov to match the shift done on the Xi
    # cov = np.hstack([cov[:, variable_index:], cov[:, :variable_index]])
    # cov = np.vstack([cov[variable_index:, :], cov[:variable_index, :]])

    # generate the znc
    znc = _generator(np.eye(cov.shape[0]), n)
    zncbis = _generator(np.eye(cov.shape[0]), n)
    zncter = np.hstack((znc[:, [0]], zncbis[:, 1:]))
    zncquad = np.hstack((zncbis[:, :-1], znc[:, [-1]]))

    # compute the L
    # print(f"eigenvals: {np.linalg.eigvals(cov)}")
    # u, s, vh = np.linalg.svd(cov, hermitian=True)
    # print(f"s: {s}")
    # cov_2 = u @ np.diag(np.clip(s, a_min=1e-8, a_max=None)) @ vh
    L = cholesky(cov)
    # compute zc
    zc = np.matmul(znc, np.matmul(inv(cholesky(_empirical_cov(znc))), L.T))
    zcbis = np.matmul(zncbis, np.matmul(inv(cholesky(_empirical_cov(zncbis))), L.T))
    zcter = np.matmul(zncter, np.matmul(inv(cholesky(_empirical_cov(zncter))), L.T))
    zcquad = np.matmul(zncquad, np.matmul(inv(cholesky(_empirical_cov(zncquad))), L.T))
    # print(f"znc{zc.var()}")
    # shift back the columns to original order
    zc = _reorder_cols(zc, variable_index, inverse=True)
    zcbis = _reorder_cols(zcbis, variable_index, inverse=True)
    zcter = _reorder_cols(zcter, variable_index, inverse=True)
    zcquad = _reorder_cols(zcquad, variable_index, inverse=True)

    # apply marginals
    zc = _apply_marginals(zc, f_inv)
    zcbis = _apply_marginals(zcbis, f_inv)
    zcter = _apply_marginals(zcter, f_inv)
    zcquad = _apply_marginals(zcquad, f_inv)

    zc = pd.DataFrame(zc, columns=orig_cols)
    zcbis = pd.DataFrame(zcbis, columns=orig_cols)
    zcter = pd.DataFrame(zcter, columns=orig_cols)
    zcquad = pd.DataFrame(zcquad, columns=orig_cols)
    # mx["zc"] = zc.mean(0)
    # print(mx)
    # print(f"zc{zc.var()}")
    # print(f"f(zc){f(np.random.standard_normal(zc.shape)).var()}")
    # compute Vhat
    V = np.mean([np.var(f(zc)), np.var(f(zcbis)), np.var(f(zcter)), np.var(f(zcquad))])
    # print(V)

    # compute sobol indices
    return np.array(
        [
            _sobol_unnormalized(zc, zcbis, zcter, f) / (V + 1e-3),
            _sobol_total_unnormalized(zcbis, zcter, f) / ((2 * V) + 1e-3),
            _sobol_ind_unnormalized(zc, zcbis, zcquad, f) / (V + 1e-3),
            _sobol_total_ind_unnormalized(zcbis, zcquad, f) / ((2 * V) + 1e-3),
        ]
    )


def _apply_marginals(z_cond, F_inv):
    """
    Apply the computed marginal in order to get back to the original distribution.

    Args:
        z_cond: the conditionally distributed gaussian vector.
        F_inv: the inverse cumulative function of each marginal

    Returns: the transformed marginals.

    """
    xj_cols = []

    for j in range(z_cond.shape[1]):
        xj_cols.append(
            np.expand_dims(F_inv[j](_phi(z_cond[:, j])), axis=1)
        )  # 2.6 step 4.
    return np.hstack(xj_cols)  # turn the list of columns into a matrix


def _compute_marginal_inv_cumul_dist(data_x: np.ndarray.__class__, N=None):
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
        cum_dists_functions.append(
            np.vectorize(
                lambda x: x_sorted[min(int(x * len(x_sorted)), len(x_sorted) - 1)]
            )
        )
    return cum_dists_functions


@np.vectorize
def _phi(x):
    """
    Cumulative distribution function for the standard normal distribution
    """
    return (1.0 + erf(x / sqrt(2.0))) / 2.0


def _reorder_cols(data, col_index, inverse=False):
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
        col_index = -col_index
    return np.hstack([data[:, col_index:], data[:, :col_index]])


def _generator(cov, n=100):
    """
    Generate normally distributed batches
    Args:
        cov: covariance matrix of the distribution
        n: number of samples

    Returns: numpy array containing the samples

    """
    return np.random.multivariate_normal([0] * np.shape(cov)[0], cov, size=n)


def _empirical_cov(data):
    """
    compute empirical covariance.
    Args:
        data: numpy array with the data.

    Returns: the covariance matrix of the data

    """
    return np.cov(np.transpose(data))


def _sobol_unnormalized(zc, zcbis, zcter, f):
    return np.mean(np.multiply(f(zc), (f(zcter) - f(zcbis))))


def _sobol_total_ind_unnormalized(zcbis, zcquad, f):
    return np.mean(np.square(f(zcquad) - f(zcbis)))


def _sobol_ind_unnormalized(zc, zcbis, zcquad, f):
    return np.mean(np.multiply(f(zc), (f(zcquad) - f(zcbis))))


def _sobol_total_unnormalized(zcbis, zcter, f):
    return np.mean(np.square(f(zcter) - f(zcbis)))
