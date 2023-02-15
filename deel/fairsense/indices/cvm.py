# -*- coding: utf-8 -*-
# Copyright IRT Antoine de Saint Exupéry et Université Paul Sabatier Toulouse III - All
# rights reserved. DEEL is a research program operated by IVADO, IRT Saint Exupéry,
# CRIAQ and ANITI - https://www.deel.ai/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import numpy as np
import pandas as pd
from sklearn.neighbors import KDTree

from deel.fairsense.utils.dataclasses import IndicesInput
from deel.fairsense.utils.dataclasses import IndicesOutput


def cvm_indices(index_input: IndicesInput) -> IndicesOutput:
    """Compute the CVM indices of a fairness problem.
    Set FairnessProblem.result as a Dataframe containing the indices.

    Warning:
        this indice may fail silently if all values of one variable are similar (
        constant ) which  may occurs when applying one hot encoding with a large
        number of splits.

    Args:
        index_input (IndicesInput): The fairness problem to study.

    Returns:
        IndicesOutput object, containing the CVM indices, one line per variable group
        and one column for each index.

    """
    # __check_arg_cvm(index_input, cols)
    df = pd.DataFrame(index_input.x, columns=index_input.x.columns)
    df["outputs"] = pd.DataFrame(index_input.compute_objective())
    return IndicesOutput(__analyze(df, "outputs", cols=index_input.variable_groups))


def __CVM(data: pd.DataFrame, x_name, z_name, y_name):
    """
    Compute the CVM index T(Y, Z|X)
    Args:
        data: the input data, already sorted by y_name, and with a column containing the
            Ri named "i"
        x_name: name of the columns in the set of variables X
        z_name: name of the columns in the set of variable Z
        y_name: name of the columns with the variable Y (not used, as we already have a
            column containing the Ri)

    Returns: the tuple(T(Y, Z), T(Y, Z|X))

    """
    # reformat the column names to have a list
    if isinstance(x_name, str):
        x_name = [x_name]
    elif isinstance(x_name, set):
        x_name = list(x_name)
    if isinstance(z_name, str):
        z_name = [z_name]
    elif isinstance(z_name, set):
        z_name = list(z_name)
    # compute Ni
    # build a KDTree containing the sampling of X
    kd_xi = KDTree(data[z_name])
    # query the tree to get the two closest neighbor of each sample. The closest will
    # be the sample itself, and the second will be the closest neighbors. This
    # returns both the distance with the neighbor and the index of the neighbor in
    # the list used to build the tree.
    dist, ind = kd_xi.query(data[z_name], k=2)
    # compute N_i ( add 1 as python indices start from 0, and the formula use indices
    # starting from 1
    data["N_i"] = ind[:, 1] + 1
    # compute M_i
    # we repeat the same process, but we work on (X,Y) this time
    kd_xiyi = KDTree(data[x_name + z_name])  # build KDTree
    # find closest neighbors
    dist, ind = kd_xiyi.query(data[x_name + z_name], k=2)
    data["M_i"] = ind[:, 1] + 1  # save M_i
    # compute L_i
    # the + 1 account the fact that Ri indices start from 1
    data["L_i"] = len(data) - data["i"]
    # compute CVM
    n = len(data)
    cvm = (
        3
        * np.mean(np.abs(data["i"].values - data["N_i"].values))  # rank of i minus
        # rank of nearest neighbor with respect to z
        / (n - 1 / n)
    )  # equation 7 from https://arxiv.org/abs/2003.01772
    # compute CVM_ind
    num_1 = np.mean(
        np.minimum(data["i"].values, data["M_i"].values)
        - np.minimum(data["i"].values, data["N_i"].values)
    )
    den_1 = np.mean(data["i"].values - np.minimum(data["i"].values, data["N_i"].values))
    num_2 = np.mean(
        (np.minimum(data["i"].values, data["N_i"].values))
        - (np.square(data["L_i"].values) / n)
    )
    den_2 = np.mean(data["L_i"].values * (1 - (data["L_i"].values / n)))
    # equations p4 of https://arxiv.org/abs/1910.12327 with translated notations
    tn_cond = num_1 / den_1
    tn_ind = num_2 / den_2
    # equation 32 appendix D of https://arxiv.org/abs/2103.04613
    cvm_ind = tn_cond * (1 - tn_ind)
    return cvm, cvm_ind


def __analyze(x, output_var, cols=None):
    """
    return the CVM indices
    Args:
        x: a dataframe containing inputs and outputs
        output_var: name of the column cotaining the output
        cols: if None one index is computed for each column,
              a list of list containing the columns names can also be provided
              then one index will be computed for each group.
              [["A"], ["B=1", "B=2"]] wil return 2 indices: one for A, and one
              for the group B.

    Returns: a Dataframe with the indices for each group.

    """
    col_was_none = cols is None
    # manipulate a copy of the dataset
    x = x.copy()
    if cols is None:
        cols = list(x.columns).copy()
        cols.remove(output_var)
    # sort the row by increasing Y
    x = x.sort_values(output_var)
    # add a columns with the rank of each row
    x["i"] = range(1, len(x) + 1)
    # commented code handle ties on the output properly ( same Y => same i )
    # rows are sorted on Y
    # compute the difference between two consecutive rows
    # x['i'] = x[output_var].rolling(2).apply(
    #     lambda x: x.index[-1] if (x.iloc[-1] != x.iloc[0]) else np.nan, # if the
    #     # difference is not null put the value of the second row, else put nan
    # )
    # x['i'].iloc[-1] = len(x['i']) # last row have index len(x)
    # x['i'] = x['i'].bfill() # fill nans with the next filled x

    indices = []
    for col in cols:
        # we compute indices for each variable
        x_names = cols.copy()
        x_names.remove(col)
        try:
            x_names = np.concatenate(x_names).tolist()
        except:
            pass
        #
        indices.append(__CVM(x.copy(), col, x_names, output_var))
        # [
        #     __CVM(x.copy(), x_names, col, output_var)[1],
        #     __CVM(x.copy(), col, x_names, output_var),
        # ]
        # )
    if col_was_none:
        index = cols
    else:
        # iterate the list take the first element and keep everything before the "="
        # this turn [["A"], ["B=1","B=2"]] into ["A", "B"]
        index = [x[0].split("=")[0] for x in cols]
    # store the outputs in a dataframe
    df = pd.DataFrame(indices, index=index, columns=["CVM", "CVM_indep"])
    return df
